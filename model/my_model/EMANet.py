from functools import partial
import math
from model.model_utils import init_weights, _FCNHead
import numpy as np
from model.segbase import SegBaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *
from ..PSPNet import _PyramidPooling
from ..DeepLabV3 import _ASPP
from .SPUnet import SPSP
from .RecoNet import Reco_module

class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        # 初始化基
        mu = torch.Tensor(1, c, k)  # k个描述子
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)  # 归一化
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        # 批中的每个图片都复制一个基
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k  # 特征图中的n个点与k个基的相似性
                z = F.softmax(z, dim=2)  # b * n * k  # 每个点属于某一个基的概率
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))  # 计算每个点对基的归一化的权重，
                mu = torch.bmm(x, z_)  # b * c * k  # 用每个点去加权组合，得到基
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation is writtern in train.py, which is significant.

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n  # 用基重建特征图
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))





class deep_conv(nn.Module):
    def __init__(self, in_channel, inter_channel,  out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True),
            nn.Conv2d(inter_channel, inter_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True),
            nn.Conv2d(inter_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        return self.conv(x)


class out_conv(nn.Module):
    def __init__(self, in_channel, n_class):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, n_class, kernel_size=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        return self.conv(x)



class EMANet(SegBaseModel):

    def __init__(self, n_class, backbone='resnet34', aux=False, pretrained_base=False, dilated=True, deep_stem=False,
                 crop_size=224,
                 **kwargs):
        super(EMANet, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated,
                                     deep_stem=deep_stem, **kwargs)
        self.aux = aux
        self.dilated = dilated
        channels = self.base_channel  # [256, 512, 1024, 2048]
        print(channels)
        if deep_stem or backbone == 'resnest101':
            conv1_channel = 128
        else:
            conv1_channel = 64

        # self.spsp = SPSP(channels[3], scales=[6, 3, 2, 1])   # scales=[6, 3, 2, 1]
        self.emau = EMAU(channels[0], k=32)

        if dilated:
            self.SF1 = AlignModule(channels[3], channels[0])
            self.donv_up3 = decoder_block(channels[0] + channels[3], channels[0])

            self.SF2 = AlignModule(channels[0], conv1_channel)
            self.donv_up4 = decoder_block(channels[0] + conv1_channel, channels[0])

            self.out_conv = out_conv(channels[0], n_class)
        else:
            # self.SF1 = AlignModule(channels[3], channels[2])
            self.donv_up1 = decoder_block(channels[2] + channels[3], channels[2])

            # self.SF2 = AlignModule(channels[2], channels[1])
            self.donv_up2 = decoder_block(channels[1] + channels[2], channels[1])

            # self.SF3 = AlignModule(channels[1], channels[0])
            self.donv_up3 = decoder_block(channels[0] + channels[1], channels[0])

            # self.SF4 = AlignModule(channels[0], channels[0])
            self.donv_up4 = decoder_block(channels[0] + conv1_channel, channels[0])
            self.out_conv = out_conv(channels[0], n_class)

        if self.aux:
            self.aux_layer = _FCNHead(channels[3], n_class)


    def forward(self, x):
        outputs = dict()
        size = x.size()[2:]

        c1, c2, c3, c4, c5 = self.backbone.extract_features(x)

        # c5 = self.spsp(c5)

        if self.dilated:
            # c5 = self.SF1((c2, c5))
            x = self.donv_up3(c5, c2)
            # x = self.SF2((c1, x))
            x = self.donv_up4(x, c1)
        else:
            # c5 = self.SF1((c4, c5))
            x = self.donv_up1(c5, c4)

            # x = self.SF2((c3, x))
            x = self.donv_up2(x, c3)

            # x = self.SF3((c2, x))
            x = self.donv_up3(x, c2)

            # x = self.SF4((c1, x))
            x = self.donv_up4(x, c1)

        x, mu = self.emau(x)

        x = self.out_conv(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 最后上采样

        outputs.update({"main_out": x})
        outputs.update({"mu": mu})

        if self.aux:
            auxout = self.aux_layer(c5)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.update({"aux_out": auxout})
        return outputs
