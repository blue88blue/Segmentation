import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_utils import init_weights, _FCNHead
from .blocks import *
from model.Unet import unet_decoder



def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x




class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)




class myInvertedResidual(nn.Module):
    def __init__(self, inchannel, dilations=[1, 2, 3]):
        super(myInvertedResidual, self).__init__()

        self.channel_group = inchannel // 4

        self.banch2 = nn.Sequential(
            # pw
            nn.Conv2d(self.channel_group, self.channel_group, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_group),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.channel_group, self.channel_group, 3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(self.channel_group),
            # pw-linear
            nn.Conv2d(self.channel_group, self.channel_group, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_group),
            nn.ReLU(inplace=True),
        )
        self.banch3 = nn.Sequential(
            # pw
            nn.Conv2d(self.channel_group, self.channel_group, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_group),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.channel_group, self.channel_group, 3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(self.channel_group),
            # pw-linear
            nn.Conv2d(self.channel_group, self.channel_group, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_group),
            nn.ReLU(inplace=True),
        )
        self.banch4 = nn.Sequential(
            # pw
            nn.Conv2d(self.channel_group, self.channel_group, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_group),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.channel_group, self.channel_group, 3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(self.channel_group),
            # pw-linear
            nn.Conv2d(self.channel_group, self.channel_group, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_group),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = x[:, :self.channel_group, :, :]
        x2 = x[:, self.channel_group:self.channel_group*2, :, :]
        x3 = x[:, self.channel_group*2:self.channel_group * 3, :, :]
        x4 = x[:, self.channel_group*3:, :, :]
        out = torch.cat((x1, self.banch2(x2), self.banch3(x3), self.banch4(x4)), dim=1)

        return channel_shuffle(out, 4)



class shuffle_Unet(nn.Module):
    def __init__(self, in_channel, n_class, aux=False, channel_reduction=2):
        super().__init__()
        self.aux = aux
        channels = [64, 128, 256, 512, 1024]
        channels = [int(c / channel_reduction) for c in channels]

        self.conv0 = conv_bn(in_channel, channels[0], stride=1)

        self.conv1 = nn.Sequential(
            myInvertedResidual(channels[0]),
            # myInvertedResidual(channels[0]),
        )
        self.conv2 = nn.Sequential(
            InvertedResidual(channels[0], channels[1], stride=2, benchmodel=2),
            myInvertedResidual(channels[1]),
            # myInvertedResidual(channels[1]),
        )
        self.conv3 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, benchmodel=2),
            myInvertedResidual(channels[2]),
            # myInvertedResidual(channels[2]),
        )
        self.conv4 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, benchmodel=2),
            myInvertedResidual(channels[3]),
            # myInvertedResidual(channels[3]),
        )

        self.conv_center = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, benchmodel=2),
            myInvertedResidual(channels[4]),
            # myInvertedResidual(channels[4]),
        )

        self.decoder1 = unet_decoder(channels[4], channels[3])
        self.decoder2 = unet_decoder(channels[3], channels[2])
        self.decoder3 = unet_decoder(channels[2], channels[1])
        self.decoder4 = unet_decoder(channels[1], channels[0])
        self.out_conv = nn.Conv2d(channels[0], n_class, kernel_size=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, x):

        x = self.conv0(x)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv_center(c4)

        x = self.decoder1(c5, c4)
        x = self.decoder2(x, c3)
        x = self.decoder3(x, c2)
        x = self.decoder4(x, c1)

        x = self.out_conv(x)
        outputs = dict()
        outputs.update({"main_out": x})
        return outputs









