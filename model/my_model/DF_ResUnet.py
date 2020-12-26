import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.segbase import SegBaseModel
from model.model_utils import init_weights, _FCNHead
from .blocks import *



class SelFuseFeature(nn.Module):
    def __init__(self, in_channels, shift_n=20, n_class=2):
        super(SelFuseFeature, self).__init__()

        self.shift_n = shift_n
        self.n_class = n_class

        self.ConvDf_1x1 = nn.Conv2d(in_channels, 2, kernel_size=1, stride=1, padding=0)
        self.fuse_conv = nn.Conv2d(in_channels * 2, n_class, kernel_size=1, padding=0)


    def forward(self, x):
        select_x = x.clone()
        N, _, H, W = x.shape

        df = self.ConvDf_1x1(x)
        df_ = df.clone()

        mag = torch.sqrt(torch.sum(df ** 2, dim=1))
        greater_mask = mag > 0.5
        greater_mask = torch.stack([greater_mask, greater_mask], dim=1)
        df[~greater_mask] = 0

        scale = 1.

        grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=0)
        grid = grid.expand(N, -1, -1, -1).to(x.device, dtype=torch.float).requires_grad_()
        grid = grid + scale * df

        grid = grid.permute(0, 2, 3, 1).transpose(1, 2)
        grid_ = grid + 0.
        grid[..., 0] = 2 * grid_[..., 0] / (H - 1) - 1
        grid[..., 1] = 2 * grid_[..., 1] / (W - 1) - 1

        # features = []

        for _ in range(self.shift_n):
            select_x = F.grid_sample(select_x, grid, mode='bilinear', padding_mode='border', align_corners=True)
            # features.append(select_x)
        # select_x = torch.mean(torch.stack(features, dim=0), dim=0)
        # features.append(select_x.detach().cpu().numpy())
        # np.save("/root/chengfeng/Cardiac/source_code/logs/acdc_logs/logs_temp/feature.npy", np.array(features))

        out = self.fuse_conv(torch.cat([x, select_x], dim=1))
        return out, df_



class SelFuseFeature_1(nn.Module):
    def __init__(self, in_channels, shift_n=5, n_class=2):
        super(SelFuseFeature_1, self).__init__()

        self.shift_n = shift_n
        self.n_class = n_class

        self.ConvDf_1x1 = nn.Conv2d(in_channels, 2, kernel_size=1, stride=1, padding=0)
        self.Conv_edge_1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.fuse_conv = nn.Conv2d(in_channels * 2, n_class, kernel_size=1, padding=0)

    def forward(self, x):
        select_x = x.clone()
        N, _, H, W = x.shape
        size = (H, W)

        edge_ = torch.sigmoid(self.Conv_edge_1x1(x))
        edge = (edge_ > 0.5).float()
        df = self.ConvDf_1x1(x)
        df_ = df.clone()

        # mag = torch.sqrt(torch.sum(df ** 2, dim=1))
        # greater_mask = mag > 0.5
        # greater_mask = torch.stack([greater_mask, greater_mask], dim=1)
        # df[~greater_mask] = 0
        df = df * edge

        norm = torch.tensor([[[[size[1], size[0]]]]]).type_as(x).to(x.device)

        flow_x = torch.linspace(-1, 1, size[1]).repeat(size[0], 1).unsqueeze(2)
        flow_y = torch.linspace(-1, 1, size[0]).view(-1, 1).repeat(1, size[1]).unsqueeze(2)
        flow = torch.cat((flow_x, flow_y), dim=2).unsqueeze(0)
        flow = flow.expand(N, -1, -1, -1).to(device=df.device, dtype=torch.float)

        grid = flow + df.permute(0, 2, 3, 1) /norm

        for _ in range(self.shift_n):
            select_x = F.grid_sample(select_x, grid, mode='bilinear', padding_mode='border', align_corners=True)

        out = self.fuse_conv(torch.cat([x, select_x], dim=1))
        return out, df_, edge_




class DF_ResUnet(SegBaseModel):

    def __init__(self, n_class, backbone='resnet34', aux=False, pretrained_base=False, dilated=True, deep_stem=False, **kwargs):
        super(DF_ResUnet, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated, deep_stem=deep_stem, **kwargs)
        self.aux = aux
        self.dilated = dilated
        channels = self.base_channel
        if deep_stem or backbone == 'resnest101':
            conv1_channel = 128
        else:
            conv1_channel = 64

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

        self.sff = SelFuseFeature_1(channels[0], n_class=n_class)


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

        if self.dilated:
            x = self.donv_up3(c5, c2)
            x = self.donv_up4(x, c1)
        else:
            x = self.donv_up1(c5, c4)
            x = self.donv_up2(x, c3)
            x = self.donv_up3(x, c2)
            x = self.donv_up4(x, c1)

        x, df, edge = self.sff(x)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 最后上采样
        df = F.interpolate(df, size, mode='bilinear', align_corners=True)  # 最后上采样
        edge = F.interpolate(edge, size, mode='bilinear', align_corners=True)  # 最后上采样
        outputs.update({"main_out": x})
        outputs.update({"df": df})
        outputs.update({"edge": edge})

        if self.aux:
            auxout = self.aux_layer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.update({"aux_out": [auxout]})
        return outputs








