""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F
from .utils import unetConv2, unetUp, conv2DBatchNormRelu, conv2DBatchNorm, init_weights
from .unet_parts import *
from .vit import ViT

class UNet(nn.Module):
    features = dict() # 记录每一层的输出
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xm = self.down4(x4)
        xu1 = self.up1(xm, x4)
        xu2 = self.up2(xu1, x3)
        xu3 = self.up3(xu2, x2)
        xu4 = self.up4(xu3, x1)
        # 经过sigmoid激活函数的输出
        logits = self.outc(xu4)

        # 记录每一层的输出
        self.features['x1'] = x1
        self.features['x2'] = x2
        self.features['x3'] = x3
        self.features['x4'] = x4
        self.features['xm'] = xm
        self.features['xu1'] = xu1
        self.features['xu2'] = xu2
        self.features['xu3'] = xu3
        self.features['xu4'] = xu4
        self.features['logits'] = logits

        return logits
        
        

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)
    # -- test --
    x = torch.randn(1, 3, 572, 572)
    logits = net(x)
    print(logits.shape)