import numpy as np
import torch
from torch import nn
from torch.nn import init



'''
在通道维度上进行注意力机制
'''
class SEAttention(nn.Module):
    # 用来返回各通道的注意力权重
    attention_weight = None
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        # 全局平均池化将每个通道的特征图转换为一个数值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        # y 即为权重矩阵 1*1*channel
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        self.attention_weight = y
        # expand_as(x) 将y扩展为x的形状
        return x * y.expand_as(x)


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    se = SEAttention(channel=512,reduction=8)
    output=se(input)
    print(output.shape)

    