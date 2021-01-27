from ..Unet import double_conv
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_utils import init_weights, _FCNHead



class Flaw_Unet(nn.Module):
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

        self.decoder1 = Unet_decoder(channels, n_class)
        self.decoder2 = Unet_decoder(channels, n_class)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, x):
        outputs = dict()

        c1 = self.donv1(x)
        c2 = self.donv2(self.down_pool(c1))
        c3 = self.donv3(self.down_pool(c2))
        c4 = self.donv4(self.down_pool(c3))
        c5 = self.donv_mid(self.down_pool(c4))

        x1 = self.decoder1([c5, c4, c3, c2, c1])
        x2 = self.decoder2([c5, c4, c3, c2, c1])

        outputs.update({"main_out": x1})
        outputs.update({"flaw_out": x2})

        return outputs



class Unet_decoder(nn.Module):
    def __init__(self, channels, n_class):
        super().__init__()
        self.donv5 = double_conv(channels[4], channels[2])
        self.donv6 = double_conv(channels[3], channels[1])
        self.donv7 = double_conv(channels[2], channels[0])
        self.donv8 = double_conv(channels[1], channels[0])
        self.out_conv = nn.Conv2d(channels[0], n_class, kernel_size=1)

    def forward(self, inputs):
        c5, c4, c3, c2, c1 = inputs

        x = F.interpolate(c5, size=c4.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv5(torch.cat((x, c4), dim=1))
        x = F.interpolate(x, size=c3.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv6(torch.cat((x, c3), dim=1))
        x = F.interpolate(x, size=c2.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv7(torch.cat((x, c2), dim=1))
        x = F.interpolate(x, size=c1.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv8(torch.cat((x, c1), dim=1))
        x = self.out_conv(x)

        return x