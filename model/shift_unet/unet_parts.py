""" Parts of the U-Net model """
"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""
import torch 
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # TODO 2d和1d的区别
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    '''Downscaling with maxpool then double conv'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    '''Upscaling then double conv'''

    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        # TODO 计算输出维度
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # TODO 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class Shift_block_up(nn.Module):
    '''shift->bn->mlp'''
    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__() 
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        # self.shift = nn.Conv2d(in_channels, out_channels, kernel_size=1) # 应该更换为shift
        # self.ln = nn.LayerNorm(out_channels)
        self.bn = nn.BatchNorm2d(in_channels // 2) # 512 -> 512 : channel
        self.mlp = nn.Sequential(
            nn.Linear(in_channels // 2, in_channels), # 512 -> 1024
            nn.ReLU(inplace=True), # 256
            nn.Linear(in_channels , in_channels // 2) # 1024 -> 512
        )
        self.conv = DoubleConv(in_channels , out_channels) # 1024(512+512) -> 256
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # x_in = self.shift(x1)
        x_in = x1
        # x = self.ln(x_in)
        x = self.bn(x_in)
        # x_in = x_in.view(x_in.size(0), -1)
        x = self.mlp(x_in)
        x_skip = x + x_in
        x = torch.cat([x2, x_skip], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    