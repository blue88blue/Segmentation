import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .model_utils import init_weights, _FCNHead


class double_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.dconv(x)
        return x



class Attention_Gate(nn.Module):
    def __init__(self, in_channel_x, in_channel_g, channel_reduce=2):
        super(Attention_Gate, self).__init__()
        self.in_channel_x = in_channel_x
        self.in_channel_g = in_channel_g
        self.inter_channel = max(in_channel_x//channel_reduce, 16)

        self.Wx = nn.Sequential(
            nn.Conv2d(in_channel_x, self.inter_channel, kernel_size=1),
            nn.BatchNorm2d(self.inter_channel)
        )
        self.Wg = nn.Sequential(
            nn.Conv2d(in_channel_g, self.inter_channel, kernel_size=1),
            nn.BatchNorm2d(self.inter_channel)
        )
        self.phi =nn.Sequential(
            nn.Conv2d(self.inter_channel, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        g_ = self.Wg(g)
        x_ = self.Wx(x)
        f = self.phi(F.relu(g_ + x_))
        x_out = f.expand_as(x) * x

        return x_out



class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x




class Att_Up(nn.Module):
    def __init__(self, channel_h, channel_l, out_channel):
        super(Att_Up, self).__init__()
        self.up = up_conv(channel_h, channel_h)
        self.AG = Attention_Gate(channel_l, channel_h)
        self.conv = double_conv(channel_h+channel_l, out_channel)

    def forward(self, x_h, x_l):
        x_h_up = self.up(x_h)
        x_l_gate = self.AG(x_l, x_h_up)
        x = torch.cat((x_l_gate, x_h_up), dim=1)
        x = self.conv(x)
        return x




class AttUnet(nn.Module):
    def __init__(self, in_channel, n_class, channel_reduction=2, aux=False):
        super().__init__()
        self.aux = aux
        channels = [64, 128, 256, 512, 1024]
        channels = [int(c / channel_reduction) for c in channels]

        self.donv1 = double_conv(in_channel, channels[0])
        self.donv2 = double_conv(channels[0], channels[1])
        self.donv3 = double_conv(channels[1], channels[2])
        self.donv4 = double_conv(channels[2], channels[3])
        self.down_pool = nn.MaxPool2d(kernel_size=2)

        self.donv_mid = double_conv(channels[3], channels[3])

        self.Up5 = Att_Up(channels[3], channels[3], channels[2])
        self.Up6 = Att_Up(channels[2], channels[2], channels[1])
        self.Up7 = Att_Up(channels[1], channels[1], channels[0])
        self.Up8 = Att_Up(channels[0], channels[0], channels[0])

        self.out_conv = nn.Conv2d(channels[0], n_class, kernel_size=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, x):
        outputs = []
        size = x.size()[2:]

        x1 = self.donv1(x)
        x2 = self.donv2(self.down_pool(x1))
        x3 = self.donv3(self.down_pool(x2))
        x4 = self.donv4(self.down_pool(x3))
        x_mid = self.donv_mid(self.down_pool(x4))

        x = self.Up5(x_mid, x4)
        x = self.Up6(x, x3)
        x = self.Up7(x, x2)
        x = self.Up8(x, x1)

        x = self.out_conv(x)
        outputs.append(x)

        return outputs



























