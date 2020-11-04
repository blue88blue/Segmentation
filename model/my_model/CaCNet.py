import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.segbase import SegBaseModel
from model.model_utils import init_weights, _FCNHead
from .blocks import *



class CaC_Module(nn.Module):
    def __init__(self, in_channel, kernel_size=3, dialted_rates=(1, 3, 5)):
        super().__init__()
        self.kernel_size = kernel_size
        self.dialted_rates = dialted_rates
        self.conv_key = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv_query = nn.Conv2d(in_channel, kernel_size ** 2, kernel_size=1)
        # self.bn = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        b, c, h, w = x.size()
        key = self.conv_key(x).reshape(b, c, h*w)
        query = self.conv_query(x).reshape(b, -1, h*w).permute(0, 2, 1)  # (b, h*w, s*s)
        query = F.softmax(query, dim=1)

        CaC_kernel = torch.bmm(key, query).reshape(b, c, self.kernel_size, self.kernel_size)  # (b, c, s, s)
        # CaC_kernel = self.bn(CaC_kernel)
        CaC_kernel = CaC_kernel.unsqueeze(2).reshape(b*c, 1, self.kernel_size, self.kernel_size)  # (b, c, 1, s, s)

        x_temp = x.reshape(b*c, h, w).unsqueeze(0)
        weight1 = F.conv2d(x_temp, CaC_kernel, groups=b*c, padding=self.dialted_rates[0], dilation=self.dialted_rates[0])
        weight2 = F.conv2d(x_temp, CaC_kernel, groups=b*c, padding=self.dialted_rates[1], dilation=self.dialted_rates[1])
        weight3 = F.conv2d(x_temp, CaC_kernel, groups=b*c, padding=self.dialted_rates[2], dilation=self.dialted_rates[2])

        weight = torch.sigmoid(weight1) + torch.sigmoid(weight2) + torch.sigmoid(weight3)  # (b*c, h, w)
        weight = weight.reshape(b, c, h, w)

        return torch.mul(x, weight)





class CaCNet(SegBaseModel):

    def __init__(self, n_class, backbone='resnet34', aux=False, pretrained_base=False, dilated=True, deep_stem=False, **kwargs):
        super(CaCNet, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated, deep_stem=deep_stem, **kwargs)
        self.aux = aux
        self.dilated = dilated
        channels = self.base_channel
        if deep_stem or backbone == 'resnest101':
            conv1_channel = 128
        else:
            conv1_channel = 64

        self.CaC1 = CaC_Module(conv1_channel)
        self.CaC2 = CaC_Module(channels[0])
        self.CaC3 = CaC_Module(channels[1])
        self.CaC4 = CaC_Module(channels[2])
        self.CaC5 = CaC_Module(channels[3])

        if dilated:
            self.donv_up3 = decoder_block(channels[0]+channels[3], channels[0])
            self.donv_up4 = decoder_block(channels[0]+conv1_channel, channels[0])
        else:
            self.donv_up1 = decoder_block(channels[2] + channels[3], channels[2])
            self.donv_up2 = decoder_block(channels[1] + channels[2], channels[1])
            self.donv_up3 = decoder_block(channels[0] + channels[1], channels[0])
            self.donv_up4 = decoder_block(channels[0] + conv1_channel, channels[0])

        if self.aux:
            self.aux_layer = _FCNHead(256, n_class)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], n_class, kernel_size=1, bias=False),
        )


    def forward(self, x):
        outputs = dict()
        size = x.size()[2:]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)  # 1/2  64
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)  # 1/4   64
        c3 = self.backbone.layer2(c2)  # 1/8   128
        c4 = self.backbone.layer3(c3)  # 1/16   256
        c5 = self.backbone.layer4(c4)  # 1/32   512

        c1 = self.CaC1(c1)
        c2 = self.CaC2(c2)
        c3 = self.CaC3(c3)
        c4 = self.CaC4(c4)
        c5 = self.CaC5(c5)

        if self.dilated:
            x = self.donv_up3(c5, c2)
            x = self.donv_up4(x, c1)
        else:
            x = self.donv_up1(c5, c4)
            x = self.donv_up2(x, c3)
            x = self.donv_up3(x, c2)
            x = self.donv_up4(x, c1)

        x = self.out_conv(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 最后上采样

        outputs.update({"main_out": x})
        if self.aux:
            auxout = self.aux_layer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.update({"auxout": [auxout]})
        return outputs













