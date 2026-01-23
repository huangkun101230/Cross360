import torch
import torch.nn as nn
import torch.nn.functional as F

def _upsample_like(src,tar):
    src = nn.functional.interpolate(src,size=tar.shape[2:],mode='bilinear',align_corners=True)
    return src

# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )

    def forward(self, x):
        out = self.deconv(x)
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip=True):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        if skip:
            self.conv_block = ConvBlock(in_channels*2, out_channels)
        else:
            self.conv_block = ConvBlock(in_channels, out_channels)
    
    def forward(self, x, skip_connection=None):
        if skip_connection is not None:
            # print("======")
            # print("x: ",x.shape)
            x = self.upconv(x)
            # print("x upconv: ",x.shape)
            # print("skip_connection: ",skip_connection.shape)
            x = torch.cat([x, skip_connection], dim=1)  # Concatenate along the channel dimension
            # print("x cat: ",x.shape)
        x = self.conv_block(x)
        # print("======")
        return x

class OutputProjDepth(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(OutputProjDepth, self).__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.proj(x)
        x = self.sigmoid(x)
        return x

class ConvDecoder(nn.Module):
    def __init__(self, max_depth=10.0):
        super(ConvDecoder, self).__init__()
        self.upconv1 = UpConvBlock(512, 256, skip=False)
        self.upconv2 = UpConvBlock(256, 128)
        self.upconv3 = UpConvBlock(128, 64)
        self.upconv4 = UpConvBlock(64, 32)
        self.upconv5 = UpConvBlock(32, 32, skip=False)

        self.output1 = OutputProjDepth(256)
        self.output2 = OutputProjDepth(128)
        self.output3 = OutputProjDepth(64)
        self.output4 = OutputProjDepth(32)
        self.output5 = OutputProjDepth(32)

        self.max_depth = nn.Parameter(torch.tensor(max_depth), requires_grad=False)
    
    def forward(self, x, encoder_features):
        e1, e2, e3, e4 = encoder_features
        x = self.upconv1(x)
        pred1 = self.output1(x)
        pred1 = self.max_depth * pred1

        x = self.upconv2(x, e3)
        pred2 = self.output2(x)
        pred2 = self.max_depth * pred2

        x = self.upconv3(x, e2)
        pred3 = self.output3(x)
        pred3 = self.max_depth * pred3

        x = self.upconv4(x, e1)
        pred4 = self.output4(x)
        pred4 = self.max_depth * pred4

        x = self.upconv5.upconv(x)
        x = self.upconv5(x)
        pred5 = self.output5(x)
        pred5 = self.max_depth * pred5

        pred4 = _upsample_like(pred4, pred5)
        pred3 = _upsample_like(pred3, pred5)
        pred2 = _upsample_like(pred2, pred5)
        pred1 = _upsample_like(pred1, pred5)
        # print('pred5: ',pred5.shape)
        # print('pred4: ',pred4.shape)
        # print('pred3: ',pred3.shape)
        # print('pred2: ',pred2.shape)
        # print('pred1: ',pred1.shape)
        # exit()
        pred_outputs = []
        pred_outputs = [pred5] + [pred4] + [pred3] + [pred2]+ [pred1]
        return pred_outputs