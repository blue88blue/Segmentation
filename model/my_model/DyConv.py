import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn.init import kaiming_normal_

def softmax_T(x, dim, T=1):

    x = torch.exp(x)
    sum = torch.sum(x, dim=dim, keepdim=True)
    x = x/sum
    return x


class attention(nn.Module):
    def __init__(self, channel, num_experts, reduction=2):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channel = math.ceil(channel / reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, num_experts, bias=False),
        )

    def forward(self, x, T):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, -1)
        y = softmax_T(y, dim=1, T=T)

        return y



class DyConv(nn.Module):
    def __init__(self, num_experts, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True):
        super().__init__()
        self.num_experts = num_experts
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.attention = attention(in_channels, num_experts, reduction=2)

        w = kaiming_normal_(torch.zeros((num_experts, out_channels, in_channels, kernel_size, kernel_size)), a=0, mode='fan_in', nonlinearity='relu')
        self.weights = nn.Parameter(w, requires_grad=True)  # 自定义的权值

        if bias == True:
            self.bias = nn.Parameter(torch.randn(num_experts, out_channels), requires_grad=True)  # 自定义的偏置
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w= x.size()
        attention_weight = self.attention(x, 1)  # （batch, num_experts）
        x_group = x.reshape(1, b*c, h, w)  # 将batch嵌入到通道中，使用分组卷积实现

        # 多个专家加权平均
        expert, outc, inc, k, k = self.weights.size()
        weight = torch.mm(attention_weight, self.weights.reshape(expert, -1)).reshape(outc * b, inc, k, k)
        bias = None
        if self.bias != None:
            bias = torch.mm(attention_weight, self.bias).view(-1)

        out = F.conv2d(x_group, weight, bias, groups=b, stride=self.stride, padding=self.padding, dilation=self.dilation)
        _, bc, ho, wo = out.size()
        out = out.reshape(b, -1, ho, wo)
        return out


if __name__ == "__main__":
    x = torch.rand((2, 1, 3, 3))
    conv = DyConv(4, 1, 2, 3)
    y = conv(x)
    print(y)

