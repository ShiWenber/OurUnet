import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from .MobileViTAttention import MobileViTAttention
from .biformer_parts.bra_nchw import nchwBRA

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class squeeze_excitation_block(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//ratio, in_channels),
            nn.Sigmoid()
        )


    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1, 1)
        return x*y.expand_as(x)

class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(in_c, out_c, kernel_size=1, padding=0)
        )

        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0, dilation=1)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(in_c, out_c, kernel_size=3, padding=18, dilation=18)

        self.c5 = Conv2D(out_c*5, out_c, kernel_size=1, padding=0, dilation=1)

    def forward(self, x):
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=x.size()[2:], mode="bilinear", align_corners=True)

        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)

        xc = torch.cat([x0, x1, x2, x3, x4], axis=1)
        y = self.c5(xc)

        return y

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, has_se: bool=True):
        super().__init__()

        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c, out_c)
        self.a1 = squeeze_excitation_block(out_c)
        self.has_se = has_se

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        if self.has_se:
            x = self.a1(x)
        return x

# 因为 vgg19 需要输入 3 通道的图片，所以启用原始的 encoder1
class encoder_vgg(nn.Module):
    def __init__(self):
        super().__init__()

        network = vgg19(pretrained=True)
        # print(network)

        self.x1 = network.features[:4]
        self.x2 = network.features[4:9]
        self.x3 = network.features[9:18]
        self.x4 = network.features[18:27]
        self.x5 = network.features[27:36]

    def forward(self, x):
        x0 = x
        x1 = self.x1(x0)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)

        return x5, [x4, x3, x2, x1]

class encoder1(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, attention: str="mobile_vit", has_se: bool=True):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))

        # self.c1 = conv_block(in_channels, 32)
        # self.c2 = conv_block(32, 64)
        # self.c3 = conv_block(64, 128)
        # self.c4 = conv_block(128, 256)
        # # 多卷积一层，能和原 vgg19 保持一致
        # self.c5 = conv_block(256, 512)

        self.c1 = conv_block(in_channels, 64, has_se=has_se)
        self.c2 = conv_block(64, 128, has_se=has_se)
        self.c3 = conv_block(128, 256, has_se=has_se)
        if attention == "mobile_vit":
            self.a = MobileViTAttention(in_channel=256, patch_size=patch_size)
        elif attention == "biformer":
            self.a = nchwBRA(256)
        elif attention == "none":
            self.a = None
        else:
            raise ValueError("attention not supported")
        self.c4 = conv_block(256, 512, has_se=has_se)
        # 多卷积一层，能和原 vgg19 保持一致
        # 多卷积的一层其实原始unet没有，似乎能提升效果？
        # self.c5 = conv_block(512, 512)





    def forward(self, x):
        x0 = x

        x1 = self.c1(x0)
        p1 = self.pool(x1)

        x2 = self.c2(p1)
        p2 = self.pool(x2)

        x3 = self.c3(p2)
        # attention
        if self.a is not None:
            x3 = self.a(x3)
        p3 = self.pool(x3)

        x4 = self.c4(p3)
        p4 = self.pool(x4)

        # x5 = self.c5(p4)

        # return p4, [x4, x3, x2, x1]

        # 多卷积一层，能和原 vgg19 保持一致
        return p4, [x4, x3, x2, x1]

class decoder1(nn.Module):
    def __init__(self, in_channels=64, skip_channels=[512, 256, 128, 64], has_se: bool=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(in_channels + skip_channels[0], 256, has_se=has_se) # 输入 x.channel + skip.channel0, 输出 256
        self.c2 = conv_block(256 + skip_channels[1], 128, has_se=has_se) # 继承上一层的输出256，输入 256 + skip.channel1, 输出 128
        self.c3 = conv_block(128 + skip_channels[2], 64, has_se=has_se)
        self.c4 = conv_block(64 + skip_channels[3], 32, has_se=has_se)

    def forward(self, x, skip):
        s1, s2, s3, s4 = skip

        x = self.up(x)
        x = torch.cat([x, s1], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, s2], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, s3], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, s4], axis=1)
        x = self.c4(x)

        return x

class encoder2(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, attention: str="mobile_vit", has_se: bool=True):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))

        self.c1 = conv_block(in_channels, 32, has_se=has_se)
        self.c2 = conv_block(32, 64, has_se=has_se)
        self.c3 = conv_block(64, 128, has_se=has_se)

        if attention == "mobile_vit":
            self.a = MobileViTAttention(in_channel=128, patch_size=patch_size)
        elif attention == "biformer":
            self.a = nchwBRA(128)
        elif attention == "none":
            self.a = None
        else:
            raise ValueError("attention not supported")
        self.c4 = conv_block(128, 256, has_se=has_se)

    def forward(self, x):
        x0 = x

        x1 = self.c1(x0)
        p1 = self.pool(x1)

        x2 = self.c2(p1)
        p2 = self.pool(x2)

        x3 = self.c3(p2)
        if self.a is not None:
            x3 = self.a(x3)
        p3 = self.pool(x3)

        x4 = self.c4(p3)
        p4 = self.pool(x4)

        return p4, [x4, x3, x2, x1]

class decoder2(nn.Module):
    def __init__(self, has_se: bool=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(832, 256, has_se=has_se)
        self.c2 = conv_block(640, 128, has_se=has_se)
        self.c3 = conv_block(320, 64, has_se=has_se)
        self.c4 = conv_block(160, 32, has_se=has_se)

    def forward(self, x, skip1, skip2):

        x = self.up(x)
        x = torch.cat([x, skip1[0], skip2[0]], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, skip1[1], skip2[1]], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, skip1[2], skip2[2]], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, skip1[3], skip2[3]], axis=1)
        x = self.c4(x)

        return x

class build_doubleunet(nn.Module):
    # def __init__(self):
    def __init__(self, in_channels=1 , num_classes=1, patch_size=4, attentions:list=["mobile_vit", "biformer"], has_se:bool = True) -> None:
        super().__init__()

        if in_channels == 3:
            self.e1 = encoder_vgg()
        else:
            self.e1 = encoder1(in_channels, patch_size=4, attention=attentions[0], has_se=has_se)
        self.a1 = ASPP(512, 64)
        self.d1 = decoder1(in_channels=64, skip_channels=[512, 256, 128, 64], has_se=has_se) # 输入 64 维度，并且将跨层连接的通道数设置为 [512, 256, 128, 64]
        self.y1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.e2 = encoder2(in_channels, patch_size=4, attention=attentions[1], has_se=has_se)
        self.a2 = ASPP(256, 64)
        self.d2 = decoder2(has_se=has_se)
        # self.y2 = nn.Conv2d(32, 1, kernel_size=1, padding=0) # 输出为 b * 1 * h * w
        self.y2 = nn.Conv2d(32, num_classes, kernel_size=1, padding=0) # 输出为 b * num_classes * h * w


    def forward(self, x):
        x0 = x # batch_size * c * h * w
        # todo 不使用5层的 encoder1改回4层
        x, skip1 = self.e1(x)  # x.channel = 512， skip1_channel = [512, 256, 128, 64]
        x = self.a1(x)
        x = self.d1(x, skip1)
        y1 = self.y1(x)

        # todo 更改 sigmoid 为其他的激活函数
        input_x = x0 * self.sigmoid(y1)
        x, skip2 = self.e2(input_x)
        x = self.a2(x)
        x = self.d2(x, skip1, skip2)
        y2 = self.y2(x)

        # print("y2: ", y2.shape)

        # 原始输出
        # return y1, y2

        # pred = y2.squeeze(dim=1)
        # print("pred: ", pred.shape)
        return y1, input_x, y2

if __name__ == "__main__":
    import numpy as np
    x = torch.randn((5, 1, 224, 224))
    model = build_doubleunet()
    y1, input2, y2 = model(x)
    # print(y1.shape, y2.shape)
    # 将y1和y2作为图片保存下来
    # 将张量转换为 NumPy 数组
    img_array1 = y1.detach().numpy()
    img_array2 = y2.detach().numpy()
    img_array3 = input2.detach().numpy()

    from PIL import Image
    img1 = Image.fromarray(np.uint8(img_array2[0, 0] * 255), mode='L')
    img2 = Image.fromarray(np.uint8(img_array2[0, 0] * 255), mode='L')
    img3 = Image.fromarray(np.uint8(img_array3[0, 0] * 255), mode='L')

    # 将图像保存到文件中
    img1.save('image.png')
    img2.save('image2.png')
    img3.save('input2.png')
    # 输入格式为 8 , 1, 256, 256