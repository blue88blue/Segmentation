import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import init_weights, _FCNHead
from .my_model.ccr import EMA_UP_docoder

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

        x = F.interpolate(x_mid, size=x4.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv5(torch.cat((x, x4), dim=1))

        x = F.interpolate(x, size=x3.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv6(torch.cat((x, x3), dim=1))

        x = F.interpolate(x, size=x2.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv7(torch.cat((x, x2), dim=1))

        x = F.interpolate(x, size=x1.size()[-2:], mode="bilinear", align_corners=True)
        x = self.donv8(torch.cat((x, x1), dim=1))

        x = self.out_conv(x)
        outputs.update({"main_out": x})

        if self.aux:
            aux_out = self.aux_layer(x_mid)
            aux_out = F.interpolate(aux_out, size, mode='bilinear', align_corners=True)
            outputs.update({"aux_out": [aux_out]})


        return outputs





if __name__ == "__main__":
    net = Unet(3, 2)

























