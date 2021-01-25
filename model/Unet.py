import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import init_weights, _FCNHead
from .my_model.ccr import EMA_UP_docoder
from .my_model.Class_GCN import class_gcn_2
from .my_model.blocks import *
from .my_model.channel_GCN import channel_gcn

class double_conv(nn.Module):
    def __init__(self, in_channel, out_channel, se=False):
        super().__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.se = se
        if se:
            self.se_layer = SELayer(out_channel)

    def forward(self, x):
        x = self.dconv(x)
        if self.se:
            x = self.se_layer(x)
        return x




class Unet(nn.Module):
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

        self.donv5 = double_conv(channels[4], channels[2])
        self.donv6 = double_conv(channels[3], channels[1])
        self.donv7 = double_conv(channels[2], channels[0])
        self.donv8 = double_conv(channels[1], channels[0])

        self.out_conv = nn.Conv2d(channels[0], n_class, kernel_size=1)

        # if aux:
        #     self.aux_layer = _FCNHead(channels[3], n_class)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, x):
        outputs = dict()
        size = x.size()[2:]

        c1 = self.donv1(x)
        c2 = self.donv2(self.down_pool(c1))
        c3 = self.donv3(self.down_pool(c2))
        c4 = self.donv4(self.down_pool(c3))
        c5 = self.donv_mid(self.down_pool(c4))

        x = F.interpolate(c5, size=c4.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv5(torch.cat((x, c4), dim=1))

        x = F.interpolate(x, size=c3.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv6(torch.cat((x, c3), dim=1))

        x = F.interpolate(x, size=c2.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv7(torch.cat((x, c2), dim=1))

        x = F.interpolate(x, size=c1.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv8(torch.cat((x, c1), dim=1))

        x = self.out_conv(x)
        outputs.update({"main_out": x})

        if self.aux:
            aux_out = self.aux_layer(c5)
            aux_out = F.interpolate(aux_out, size, mode='bilinear', align_corners=True)
            outputs.update({"aux_out": [aux_out]})


        return outputs



class Unet_upconv(nn.Module):
    def __init__(self, in_channel, n_class, channel_reduction=2, aux=False):
        super().__init__()
        self.aux = aux
        channels = [64, 128, 256, 512, 1024]
        channels = [int(c / channel_reduction) for c in channels]

        self.donv1 = double_conv(in_channel, channels[0])
        self.donv2 = double_conv(channels[0], channels[1])
        self.donv3 = double_conv(channels[1], channels[2])
        self.donv4 = double_conv(channels[2], channels[3])
        self.donv_mid = double_conv(channels[3], channels[4])

        self.down_pool = nn.MaxPool2d(kernel_size=2)

        self.donv5 = unet_decoder(channels[4], channels[3])
        self.donv6 = unet_decoder(channels[3], channels[2])
        self.donv7 = unet_decoder(channels[2], channels[1])
        self.donv8 = unet_decoder(channels[1], channels[0])

        self.out_conv = nn.Conv2d(channels[0], n_class, kernel_size=1)

        if aux:
            self.aux_layer = _FCNHead(channels[3], n_class)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, x):
        outputs = dict()
        size = x.size()[2:]

        x1 = self.donv1(x)
        x2 = self.donv2(self.down_pool(x1))
        x3 = self.donv3(self.down_pool(x2))
        x4 = self.donv4(self.down_pool(x3))
        x_mid = self.donv_mid(self.down_pool(x4))

        x = self.donv5(x_mid, x4)
        x = self.donv6(x, x3)
        x = self.donv7(x, x2)
        x = self.donv8(x, x1)
        x = self.out_conv(x)
        outputs.update({"main_out": x})

        if self.aux:
            aux_out = self.aux_layer(x_mid)
            aux_out = F.interpolate(aux_out, size, mode='bilinear', align_corners=True)
            outputs.update({"aux_out": [aux_out]})
        return outputs



class unet_decoder(nn.Module):
    def __init__(self, channel_h, channel_l):
        super().__init__()
        self.d_conv = double_conv(channel_l*2, channel_l)
        self.upconv = nn.Conv2d(channel_h, channel_l, 1)

    def forward(self, x_h, x_l):
        x_h_up = F.interpolate(x_h, size=x_l.size()[-2:], mode="bilinear", align_corners=True)
        x_h_up = self.upconv(x_h_up)
        x = torch.cat((x_h_up, x_l), dim=1)
        return self.d_conv(x)




if __name__ == "__main__":
    net = Unet(3, 2)

























