from Conversion.equi2pers_v3 import equi2pers
import torch
import torch.nn as nn
from Networks.module.blocks import Transformer_Block 
import copy
import cv2
import numpy as np

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

class TP_Embeding(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, patch_size=8, nrows=6, fov=60):
        super(TP_Embeding, self).__init__()
        self.nrows = nrows
        # self.npatches = npatches
        self.patch_size = patch_size
        self.fov = fov
        # self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, bias=True)
        self.proj = nn.Linear(input_dim*patch_size**2, output_dim)

        # self.transformer = Transformer_cascade(emb_dims=dim, num_patch=npatches, depth=6, num_heads=4)
        
    def forward(self, x):
        # B,C,H,W = x.shape
        tangent_patch = equi2pers(x, self.fov, self.nrows, patch_size=self.patch_size)
        # print("tangent_patch: ",tangent_patch.shape)
        # tangent_patch = torch.cat((tangent_patch,tangent_patch),dim=1)
        # print("tangent_patch: ",tangent_patch.shape)
        emb = tangent_patch.permute(0,4,1,2,3).flatten(2)
        # emb = tangent_patch.permute(0,4,1,2,3).reshape(B*self.npatches,C,32,32)
        # print("emb: ", emb.shape)
        emb = self.proj(emb)
        # print("emb: ", emb.shape)

        # exit()
        # print("emb: ", emb.shape)
        # emb = emb.reshape(B, self.npatches, 512)
        # print("emb: ", emb.shape)
        # emb = self.transformer(emb)
        # print("emb: ", emb.shape)
        # exit()
        return emb

        # erp = pers2equi(tangent_patch, fov=self.fov, nrows=self.nrows, patch_size=self.patch_size, erp_size=(256, 512), layer_name='test')
        # img_erp_int = erp[0, ...].permute(1, 2, 0).numpy()
        # img_erp_int = img_erp_int #* 255
        # img_erp_int = img_erp_int.astype(np.uint8)
        # print("erp: ", erp.shape)
        # print("erp: ", erp.shape)
        # exit()
        # imagenet_mean = np.array([0.485, 0.456, 0.406])
        # imagenet_std = np.array([0.229, 0.224, 0.225])
        # erp = erp.permute(0,2,3,1).detach().cpu()
        # erp = torch.clip((erp * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()

        # for i in range(4):
        #     im = erp[i]
        #     im = im[:,:,::-1]
        #     cv2.imwrite(str(i)+".png", im)
            
        # exit()