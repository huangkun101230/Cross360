from Networks.resnet_encoder import ResNet34Multiscale as Conv_Encoder
from Networks.vit_encoder import TP_Embeding
import torch
import torch.nn as nn
import torch.nn.functional as F
from Conversion.pers2equi_v3 import pers2equi
from utils.switchable_norm import SwitchNorm1d
from Networks.module.blocks import Transformer_Block 
import copy
from Networks.conv_decoder import ConvBlock, OutputProjDepth, Upsample, _upsample_like


class AttentionBetweenScales(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(AttentionBetweenScales, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ProgressiveFeatureAggregationWithAttention(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super(ProgressiveFeatureAggregationWithAttention, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.attention_between_scales = nn.ModuleList()
        
        for i in range(len(in_channels_list)):
            self.conv_layers.append(nn.Conv2d(in_channels_list[i], out_channels_list[i], kernel_size=3, padding=1))
            self.attention_between_scales.append(AttentionBetweenScales(out_channels_list[i]))
        # print(self.attention_between_scales)
        # exit()
    def forward(self, features):
        # Start with the smallest scale
        aggregated_features = self.conv_layers[-1](features[-1])
        aggregated_features = self.attention_between_scales[-1](aggregated_features)
        # Progressively aggregate features from higher scales with attention
        for i in range(len(features) - 2, -1, -1):
            # print("aggregated_features: ",aggregated_features.shape)
            # print(features[i].shape[-2:])
            upsampled_features = F.interpolate(aggregated_features, size=features[i].shape[-2:], mode='nearest')
            # print("upsampled_features: ",upsampled_features.shape)
            # print("features[i]: ",features[i].shape)
            combined_features = features[i] + upsampled_features
            aggregated_features = self.conv_layers[i](combined_features)
            aggregated_features = self.attention_between_scales[i](aggregated_features)
        
        return aggregated_features
        
class Transformer_cascade(nn.Module):
    def __init__(self, emb_dims, num_patch, depth, num_heads):
        super(Transformer_cascade, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(emb_dims, eps=1e-6)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patch, emb_dims))
        nn.init.trunc_normal_(self.pos_emb, std=.02)
        for _ in range(depth):
            layer = Transformer_Block(emb_dims, num_heads=num_heads)
            self.layer.append(copy.deepcopy(layer))
        # print(self.layer)
    def forward(self, x):
        hidden_states = x + self.pos_emb
        for i, layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states)
            
        encoded = self.encoder_norm(hidden_states)
        return encoded      
    
class CrossAttention(nn.Module):
    def __init__(self, dim=512):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Linear(dim, dim // 8)
        self.key_conv = nn.Linear(dim, dim // 8)
        self.value_conv = nn.Linear(dim, dim)
        self.q_switch_norm = SwitchNorm1d(dim)
        self.k_switch_norm = SwitchNorm1d(dim)
        self.v_switch_norm = SwitchNorm1d(dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.xavier_uniform_(self.value_conv.weight)

    def forward(self, Q, K, V):
        # Linear transformations
        query = self.query_conv(self.q_switch_norm(Q))  # (batch_size, num_queries, dim // 8)
        key = self.key_conv(self.k_switch_norm(K))      # (batch_size, num_keys, dim // 8)
        value = self.value_conv(self.v_switch_norm(V))  # (batch_size, num_keys, dim)
        # print("query: ",query.shape)
        # print("key: ",key.shape)
        # Attention mechanism
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # (batch_size, num_queries, num_keys)
        # print("attention_scores: ",attention_scores.shape)
        attention_scores = F.softmax(attention_scores, dim=-1)
        # print("attention_scores: ",attention_scores.shape)
        
        out = torch.bmm(attention_scores, value)  # (batch_size, num_queries, dim)
        # print("out: ",out.shape)
        out = out + Q  # Residual connection
        # print("out: ",out.shape)
        return out

"""Cross Projection Feature Allignment"""
class CPFA(nn.Module):
    def __init__(self, dim=512, fov=60, nrows=6, num_patch=46, patch_size=4, erp_h=8, erp_w=16, layername="CPFA"):
        super(CPFA, self).__init__()
        self.fov = fov
        self.nrows = nrows
        self.ps = patch_size
        self.erp_h = erp_h
        self.erp_w = erp_w
        self.layername = layername
        self.cross_atten = CrossAttention(dim=dim)
        self.dim_transform = nn.Linear(dim, dim*patch_size**2)
        self.self_atten = Transformer_cascade(emb_dims=dim, num_patch=num_patch, depth=2, num_heads=4)

    def forward(self, tp_emb, conv_emb):
        vit_emb = self.self_atten(tp_emb)
        # print("vit_emb: ",vit_emb.shape)
        conv_emb = conv_emb.flatten(2).transpose(1, 2)
        feat = self.cross_atten(vit_emb, conv_emb, conv_emb)
        # print("feat: ",feat.shape)
        B,L,C = feat.shape
        feat = self.dim_transform(feat)
        # print("feat: ",feat.shape)
        feat = feat.reshape(B,L,self.ps,self.ps,C).permute(0,4,2,3,1)
        # print("feat: ",feat.shape)
        feat = pers2equi(feat, self.fov, self.nrows, (self.ps, self.ps), (self.erp_h, self.erp_w), self.layername)      
        # print("feat: ",feat.shape)
        return feat
    
class OursNetwork(nn.Module):
    def __init__(self, max_depth = 10.0, patch_size=4, fov=72, nrows=5, num_patch=26):
        super().__init__()
        self.ResnetEncoder = Conv_Encoder()
        self.TPlayer_0 = TP_Embeding(input_dim=3, output_dim=512, fov=fov, nrows=nrows, patch_size=patch_size)
        self.TPlayer_1 = TP_Embeding(input_dim=256, output_dim=256, fov=fov, nrows=nrows, patch_size=patch_size*2)
        self.TPlayer_2 = TP_Embeding(input_dim=128, output_dim=128, fov=fov, nrows=nrows, patch_size=patch_size*4)
        self.TPlayer_3 = TP_Embeding(input_dim=64, output_dim=64, fov=fov, nrows=nrows, patch_size=patch_size*6)

        self.CPFAlayer_0 = CPFA(dim=512, fov=fov, nrows=nrows, num_patch=num_patch, patch_size=patch_size, erp_h=16, erp_w=32, layername="CPFA_0")
        self.CPFAlayer_1 = CPFA(dim=256, fov=fov, nrows=nrows, num_patch=num_patch, patch_size=patch_size*2, erp_h=32, erp_w=64, layername="CPFA_1")
        self.CPFAlayer_2 = CPFA(dim=128, fov=fov, nrows=nrows, num_patch=num_patch, patch_size=patch_size*4, erp_h=64, erp_w=128, layername="CPFA_2")
        self.CPFAlayer_3 = CPFA(dim=64, fov=fov, nrows=nrows, num_patch=num_patch, patch_size=patch_size*6, erp_h=128, erp_w=256, layername="CPFA_3")
        
        self.upsample_0 = Upsample(512, 256)
        self.upsample_1 = Upsample(256, 128)
        self.upsample_2 = Upsample(128, 64)
        self.upsample_3 = Upsample(64, 64)

        self.decoder_0 = ConvBlock(512*2, 512)
        self.decoder_1 = ConvBlock(256*2, 256)
        self.decoder_2 = ConvBlock(128*2, 128)
        self.decoder_3 = ConvBlock(64*2, 64)
        self.decoder_4 = ConvBlock(64*2, 64)

        self.output0 = OutputProjDepth(512)
        self.output1 = OutputProjDepth(256)
        self.output2 = OutputProjDepth(128)
        self.output3 = OutputProjDepth(64)
        self.output4 = OutputProjDepth(64)

        self.max_depth = nn.Parameter(torch.tensor(max_depth), requires_grad=False)

        self.pfaa = ProgressiveFeatureAggregationWithAttention(in_channels_list=[64,64,128,256,512], out_channels_list=[64,64,64,128,256])
        # exit()
    def forward(self, x):
        import time
        # print("input: ",x.shape)
        # start_time = time.time()
        conv_feats = self.ResnetEncoder(x)
        # end_time = time.time()
        # print(f"Elapsed time: {end_time-start_time} seconds")

        # for i in conv_feats:
        #     print(i.shape)
        # start_time = time.time()

        tp_feat0 = self.TPlayer_0(x)
        # print("tp_feat0: ",tp_feat0.shape)
        # exit()
        feat0 = self.CPFAlayer_0(tp_feat0, conv_feats[-1])
        de_feat0 = torch.cat([feat0, conv_feats[-1]], dim=1)
        de_feat0 = self.decoder_0(de_feat0)
        # print("de_feat0: ",de_feat0.shape)
        pred0 = self.output0(de_feat0)
        pred0 = self.max_depth * pred0
        # print("pred0: ",pred0.shape)
        # end_time = time.time()
        # print(f"Elapsed time: {end_time-start_time} seconds")

        # start_time = time.time()

        de_feat1 = self.upsample_0(de_feat0)
        # print("de_feat1: ",de_feat1.shape)
        tp_feat1 = self.TPlayer_1(de_feat1)
        # print("tp_feat1: ",tp_feat1.shape)
        feat1 = self.CPFAlayer_1(tp_feat1, conv_feats[-2])
        de_feat1 = torch.cat([feat1, conv_feats[-2]], dim=1)
        de_feat1 = self.decoder_1(de_feat1)
        pred1 = self.output1(de_feat1)
        pred1 = self.max_depth * pred1
        # print("pred1: ",pred1.shape)
        # end_time = time.time()
        # print(f"Elapsed time: {end_time-start_time} seconds")

        # start_time = time.time()

        de_feat2 = self.upsample_1(de_feat1)
        tp_feat2 = self.TPlayer_2(de_feat2)
        feat2 = self.CPFAlayer_2(tp_feat2, conv_feats[-3])
        de_feat2 = torch.cat([feat2, conv_feats[-3]], dim=1)
        de_feat2 = self.decoder_2(de_feat2)
        pred2 = self.output2(de_feat2)
        pred2 = self.max_depth * pred2
        # print("pred2: ",pred2.shape)
        # end_time = time.time()
        # print(f"Elapsed time: {end_time-start_time} seconds")

        # start_time = time.time()

        de_feat3 = self.upsample_2(de_feat2)
        tp_feat3 = self.TPlayer_3(de_feat3)
        feat3 = self.CPFAlayer_3(tp_feat3, conv_feats[-4])
        de_feat3 = torch.cat([feat3, conv_feats[-4]], dim=1)
        de_feat3 = self.decoder_3(de_feat3)
        pred3 = self.output3(de_feat3)
        pred3 = self.max_depth * pred3
        # print("pred3: ",pred3.shape)
        # end_time = time.time()
        # print(f"Elapsed time: {end_time-start_time} seconds")

        # start_time = time.time()

        de_feat4 = self.upsample_3(de_feat3)
        # tp_feat4 = self.TPlayer_4(de_feat4)
        # feat4 = self.CPFAlayer_4(tp_feat4, conv_feats[-5])
        de_feat4 = torch.cat([de_feat4, conv_feats[-5]], dim=1)
        de_feat4 = self.decoder_4(de_feat4)
        # pred4 = self.output4(de_feat4)
        # pred4 = self.max_depth * pred4
        # pred4 = F.interpolate(pred4, scale_factor=2, mode='nearest')

        # print("de_feat0: ",de_feat0.shape)
        # print("de_feat1: ",de_feat1.shape)
        # print("de_feat2: ",de_feat2.shape)
        # print("de_feat3: ",de_feat3.shape)
        # print("de_feat4: ",de_feat4.shape)

        pfaa_feat = self.pfaa([de_feat4, de_feat3, de_feat2, de_feat1, de_feat0])
        # print("pfaa_feat: ",pfaa_feat.shape)
        pred4 = self.output4(pfaa_feat)
        pred4 = self.max_depth * pred4

        # print("pred4: ",pred4.shape)
        # end_time = time.time()
        # print(f"Elapsed time: {end_time-start_time} seconds")
        # exit()
        # start_time = time.time()

        pred3 = _upsample_like(pred3, pred4)
        pred2 = _upsample_like(pred2, pred4)
        pred1 = _upsample_like(pred1, pred4)
        pred0 = _upsample_like(pred0, pred4)
        # end_time = time.time()
        # print(f"Elapsed time: {end_time-start_time} seconds")

        pred_outputs = []
        pred_outputs = [pred4] + [pred3] + [pred2]+ [pred1]+ [pred0]

        # exit()
        # print("ffca_feat: ",ffca_feat.shape)
        # pred = self.conv_decoder(ffca_feat, conv_feat)
        # print("pred: ",pred.shape)
        # exit()
        outputs = {}
        outputs["pred_depth"] = pred_outputs[0]
        outputs["pred_multiscale_depth"] = pred_outputs

        return outputs
