import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.segbase import SegBaseModel
from model.model_utils import init_weights, _FCNHead
from .blocks import *
from model.my_model.GloRe import GCN, GloRe_Unit_2D



class GloRe_Unit(nn.Module):
    """
    Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    """

    def __init__(self, num_in, num_mid,
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False):
        super(GloRe_Unit, self).__init__()

        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        # reduce dim
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04)  # should be zero initialized

    def forward(self, x):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        # out = x + self.blocker(self.conv_extend(x_state))

        return self.blocker(self.conv_extend(x_state))





class class_GCN(nn.Module):
    def __init__(self, in_channel, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv = _FCNHead(in_channel, n_class)

        self.GloRe = nn.ModuleList()
        for i in range(n_class):
            self.GloRe.append(GloRe_Unit(in_channel, in_channel//4, ConvNd=nn.Conv2d, BatchNormNd=nn.BatchNorm2d,))

    def forward(self, x):
        idn = x
        aux_out = self.conv(x)
        aux_pred = F.softmax(aux_out, dim=1)

        gcn_out = []
        for i in range(self.n_class):
            x_i = x * aux_pred[:, i, ...].unsqueeze(dim=1)
            gcn_out.append(self.GloRe[i](x_i))

        y = sum(gcn_out) + idn
        return aux_out, y



class squeeze_and_expand(nn.ModuleList):
    def __init__(self, in_channel, num_node, num_inter_channel,
                 ConvNd=nn.Conv2d,
                 BatchNormNd=nn.BatchNorm2d,
                 normalize=False):
        super(squeeze_and_expand, self).__init__()

        self.normalize = normalize
        self.num_s = int(1 * num_inter_channel)
        self.num_n = int(1 * num_node)

        # reduce dim
        self.conv_state = ConvNd(in_channel, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(in_channel, self.num_n, kernel_size=1)

        # extend dimension
        self.conv_extend = ConvNd(self.num_s, in_channel, kernel_size=1, bias=False)
        self.blocker = BatchNormNd(in_channel, eps=1e-04)  # should be zero initialized

    def squeeze(self, x):
        self.n = x.size(0)
        self.hw_size = x.size()[2:]

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(self.n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(self.n, self.num_n, -1)
        # x_proj_reshaped = F.softmax(x_proj_reshaped, dim=-1)  ########################################################3
        self.x_rproj_reshaped = x_proj_reshaped

        # projection: coordinate space -> interactio./n space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        return x_n_state, x_proj_reshaped.view(self.n, self.num_n, self.hw_size[0], -1)

    def expand(self, x_n_state):
        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_state, self.x_rproj_reshaped)

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(self.n, self.num_s, *self.hw_size)

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        # out = x + self.blocker(self.conv_extend(x_state))

        return self.blocker(self.conv_extend(x_state))


class class_gcn_2(nn.Module):
    def __init__(self, in_channel, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv = _FCNHead(in_channel, n_class)

        inter_channel = in_channel//2
        self.node = 64
        self.st1 = squeeze_and_expand(in_channel, self.node, inter_channel)
        self.st2 = squeeze_and_expand(in_channel, self.node, inter_channel)
        self.gcn = GCN(inter_channel, int(self.node*2))

    def forward(self, x, aux_pred=None):
        idn = x
        aux_out = None
        if aux_pred is None:
            aux_out = self.conv(x)
            aux_pred = F.softmax(aux_out, dim=1)
            aux_pred = aux_pred.detach()

        x1 = x * aux_pred[:, 0, ...].unsqueeze(dim=1)
        x2 = x * aux_pred[:, 1, ...].unsqueeze(dim=1)
        x_n_state_1, x_proj_1 = self.st1.squeeze(x1)
        x_n_state_2, x_proj_2 = self.st2.squeeze(x2)
        x_n_state = torch.cat((x_n_state_1, x_n_state_2), dim=-1)

        x_n_rel = self.gcn(x_n_state)  # (b, c, node*2)
        x_n_rel_1, x_n_rel_2 = x_n_rel[:, :, :self.node], x_n_rel[:, :, self.node:]

        x1_restruct = self.st1.expand(x_n_rel_1)
        x2_restruct = self.st2.expand(x_n_rel_2)
        out = idn + x1_restruct + x2_restruct
        return {"aux_out": aux_out,
                "out": out,
                "aux_pred": aux_pred}


class class_gcn_3(nn.Module):
    def __init__(self, in_channel, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv = _FCNHead(in_channel, n_class)

        inter_channel = in_channel//2
        self.node = 64
        self.st1 = squeeze_and_expand(in_channel, self.node, inter_channel)
        self.st2 = squeeze_and_expand(in_channel, self.node, inter_channel)
        self.gcn = GCN(inter_channel, int(self.node*2))

        self.mse = nn.MSELoss()

    def forward(self, x):
        b, c, h, w = x.size()
        idn = x
        aux_out = self.conv(x)
        aux_pred = F.softmax(aux_out, dim=1)
        aux_pred = aux_pred.detach()

        x1 = x * aux_pred[:, 0, ...].unsqueeze(dim=1)
        x2 = x * aux_pred[:, 1, ...].unsqueeze(dim=1)
        x_n_state_1 = self.st1.squeeze(x1)
        x_n_state_2 = self.st2.squeeze(x2)
        x_n_state = torch.cat((x_n_state_1, x_n_state_2), dim=-1)

        x_n_rel = self.gcn(x_n_state)  # (b, c, node*2)
        x_n_rel_1, x_n_rel_2 = x_n_rel[:, :, :self.node], x_n_rel[:, :, self.node:]

        # 相似度损失计算
        x_n_rel_norm = self._l2norm(x_n_rel, dim=1)
        x_n_rel_norm_T = x_n_rel_norm.permute(0, 2, 1)  # (b, node*2, c)
        similarity = torch.bmm(x_n_rel_norm_T, x_n_rel_norm)  # (b, node*2, node*2)
        sim_target = torch.zeros((b, int(self.node*2), int(self.node*2))).cuda()
        sim_target[:, :self.node, :self.node] = 1
        sim_target[:, self.node:, :self.node:] = 1
        sim_loss = self.mse(similarity, sim_target)

        x1_restruct = self.st1.expand(x_n_rel_1)
        x2_restruct = self.st2.expand(x_n_rel_2)
        out = idn + x1_restruct + x2_restruct
        return aux_out, out, sim_loss

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))





class class_gcn_Net(SegBaseModel):

    def __init__(self, n_class, backbone='resnet34', aux=False, pretrained_base=False, dilated=True, deep_stem=False, **kwargs):
        super(class_gcn_Net, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated, deep_stem=deep_stem, **kwargs)
        self.aux = aux
        self.dilated = dilated
        channels = self.base_channel
        if deep_stem or backbone == 'resnest101':
            conv1_channel = 128
        else:
            conv1_channel = 64

        self.class_gcn1 = class_gcn_2(channels[3], n_class)
        self.class_gcn2 = class_gcn_2(channels[2], n_class)
        self.class_gcn3 = class_gcn_2(channels[1], n_class)
        self.class_gcn4 = class_gcn_2(channels[0], n_class)

        if dilated:
            self.donv_up3 = decoder_block(channels[0]+channels[3], channels[0])
            self.donv_up4 = decoder_block(channels[0]+conv1_channel, channels[0])
        else:
            self.donv_up1 = decoder_block(channels[2] + channels[3], channels[2])
            self.donv_up2 = decoder_block(channels[1] + channels[2], channels[1])
            self.donv_up3 = decoder_block(channels[0] + channels[1], channels[0])
            self.donv_up4 = decoder_block(channels[0] + conv1_channel, channels[0])

        self.out_conv = nn.Sequential(
            # nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(channels[0]),
            # nn.ReLU(),
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

        out_gcn = self.class_gcn1(c5)
        c5 = out_gcn["out"]
        aux_pred = out_gcn["aux_pred"]
        aux_out = []
        aux_out.append(F.interpolate(out_gcn["aux_out"], size, mode='bilinear', align_corners=True))

        c4 = self.class_gcn2(c4, F.interpolate(aux_pred, c4.size()[-2:], mode='bilinear', align_corners=True))["out"]
        c3 = self.class_gcn3(c3, F.interpolate(aux_pred, c3.size()[-2:], mode='bilinear', align_corners=True))["out"]
        c2 = self.class_gcn4(c2, F.interpolate(aux_pred, c2.size()[-2:], mode='bilinear', align_corners=True))["out"]

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
        outputs.update({"aux_out": aux_out})
        return outputs







