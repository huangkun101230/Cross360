import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Define a class to extract intermediate features from ResNet-34
class ResNet34Multiscale(nn.Module):
    def __init__(self):
        super(ResNet34Multiscale, self).__init__()
        original_model = models.resnet34(pretrained=True)

        self.enc_block0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )

        # Extract specific layers from ResNet-34
        self.conv1 = nn.Sequential(*list(original_model.children())[:3])
        # print(self.conv1)
        # exit()
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        features = []
        x0 = self.enc_block0(x)
        features.append(x0)
        x = self.conv1(x)
        # print(x.shape)
        # print(x0.shape)
        # exit()
        x = self.layer1(x)
        features.append(x)
        # print(x.shape)
        # print(x.shape)
        # exit()
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        # for i in features:
        #     print(i.shape)
        # exit()
        return features