import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.segbase import SegBaseModel
from model.model_utils import init_weights, _FCNHead
from .blocks import *
from model.my_model.GloRe import GCN, GloRe_Unit_2D




class channel_gcn(nn.Module):

    def __init__(self, in_channel, group):
        super().__init__()
        assert in_channel % group == 0
        self.group_size = in_channel//group

        self.conv_mask = nn.Conv2d(in_channel, group, kernel_size=3, padding=1, bias=False)
        self.gcn = GCN(num_state=self.group_size, num_node=group)

    def forward(self, x):
        b, c, h, w = x.size()
        mask = self.conv_mask(x).view(b, -1, h*w)
        mask = F.softmax(mask, dim=-1).contiguous()  # (b, g, h, w)

        # 映射到通道相关的节点
        x_split = torch.split(x.view(b, c, -1), split_size_or_sections=self.group_size, dim=1)  # (b, c, hw)
        mask_split = torch.split(mask, split_size_or_sections=1, dim=1)  # (b, g, hw)
        temp = []
        for i in range(len(x_split)):
            temp.append(torch.bmm(x_split[i],  mask_split[i].permute(0, 2, 1)))
        nodes = torch.cat(temp, dim=-1)  # (b, c_, g)

        # GCN 得到通道注意力
        nodes = self.gcn(nodes)
        v = nodes.view(b, -1, 1, 1)

        return x+v




class channel_gcn_Net(SegBaseModel):

    def __init__(self, n_class, backbone='resnet34', aux=False, pretrained_base=False, dilated=True, deep_stem=False, **kwargs):
        super(channel_gcn_Net, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated, deep_stem=deep_stem, **kwargs)
        self.aux = aux
        self.dilated = dilated
        channels = self.base_channel
        if deep_stem or backbone == 'resnest101':
            conv1_channel = 128
        else:
            conv1_channel = 64


        self.chGCN1 = channel_gcn(channels[3], 16)
        self.chGCN2 = channel_gcn(channels[2], 16)
        self.chGCN3 = channel_gcn(channels[1], 16)
        self.chGCN4 = channel_gcn(channels[0], 16)
        self.chGCN5 = channel_gcn(conv1_channel, 16)

        if dilated:
            self.donv_up3 = decoder_block(channels[0]+channels[3], channels[0])
            self.donv_up4 = decoder_block(channels[0]+conv1_channel, channels[0])
        else:
            self.donv_up1 = decoder_block(channels[2] + channels[3], channels[2])
            self.donv_up2 = decoder_block(channels[1] + channels[2], channels[1])
            self.donv_up3 = decoder_block(channels[0] + channels[1], channels[0])
            self.donv_up4 = decoder_block(channels[0] + conv1_channel, channels[0])

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], n_class, kernel_size=1),
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

        c5 = self.chGCN1(c5)

        if self.dilated:
            x = self.donv_up3(c5, c2)
            x = self.donv_up4(x, c1)
        else:
            x = self.donv_up1(c5, c4)
            x = self.chGCN2(x)
            x = self.donv_up2(x, c3)
            x = self.chGCN3(x)
            x = self.donv_up3(x, c2)
            x = self.chGCN4(x)
            x = self.donv_up4(x, c1)
            x = self.chGCN5(x)

        x = self.out_conv(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 最后上采样

        outputs.update({"main_out": x})

        return outputs
































