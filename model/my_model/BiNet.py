import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.segbase import SegBaseModel
from model.model_utils import init_weights, _FCNHead
from .blocks import *
from model.my_model.Class_GCN import class_gcn_2
from model.my_model.ccr import EM


# EMUP
class BiNet(nn.Module):

    def __init__(self, n_class, backbone='resnet34', aux=False, inchannel=3,  pretrained_base=False, **kwargs):
        super().__init__()
        self.aux = aux
        self.detail_branch = Detail_Branch(inchannel, 128)
        self.sematic_branch = Semantic_Branch(n_class, aux=aux, backbone=backbone, pretrained_base=pretrained_base, **kwargs)

        self.se_deatil = SELayer(128, reduction=4)
        self.se_sematic = SELayer(self.sematic_branch.base_channel[-1], reduction=4)

        self.effcient_module = EMA_UP_docoder_EM_x(self.sematic_branch.base_channel[-1], 128, n_class=n_class)


    def forward(self, x):
        outputs = dict()
        x_detail = self.se_deatil(self.detail_branch(x))
        sematic_out = self.sematic_branch(x)
        x_sematic = self.se_sematic(sematic_out["main_out"])

        last_out = self.effcient_module(x_sematic, x_detail)
        main_out = F.interpolate(last_out["out"], size=x.size()[-2:], mode="bilinear", align_corners=True)

        outputs.update({"main_out": main_out})
        outputs.update({"mu": last_out["base"]})
        if self.aux:
            outputs.update({"aux_out": sematic_out["aux_out"]})
        return outputs




class BiNet_baseline(nn.Module):

    def __init__(self, n_class, backbone='resnet34', aux=False, inchannel=3,  pretrained_base=False, **kwargs):
        super().__init__()
        self.aux = aux
        self.detail_branch = Detail_Branch(inchannel, 128)
        self.sematic_branch = Semantic_Branch(n_class, aux=aux, backbone=backbone, pretrained_base=pretrained_base, **kwargs)

        # self.cg_detail = class_gcn_2(128, n_class)
        # self.cg_sematic = class_gcn_2(self.sematic_branch.base_channel[-1], n_class)

        self.out_conv = out_conv(128 + self.sematic_branch.base_channel[-1], n_class)


    def forward(self, x):
        outputs = dict()
        x_detail = self.detail_branch(x)
        x_sematic = self.sematic_branch(x)["main_out"]

        x_sematic_up = F.interpolate(x_sematic, size=x_detail.size()[-2:], mode="bilinear", align_corners=True)
        main_out = self.out_conv(torch.cat((x_sematic_up, x_detail), dim=1))
        main_out = F.interpolate(main_out, size=x.size()[-2:], mode="bilinear", align_corners=True)

        outputs.update({"main_out": main_out})
        return outputs






class EMA_UP_docoder_EM_x(nn.Module):
    def __init__(self, channel_h, channel_l, n_class, k=64):
        super().__init__()
        self.channel = channel_h + channel_l

        # self.conv_in_h = conv_bn_relu(channel_h, channel_h, kernel_size=1, padding=0)
        # self.conv_in_l = conv_bn_relu(channel_l, channel_l, kernel_size=1, padding=0)

        self.em = EM(self.channel, k=k)

        self.conv_trans_low = nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_trans_attention = nn.Conv2d(self.channel, self.channel, kernel_size=1)

        self.rebuild_conv = nn.Sequential(
                        nn.Conv2d(self.channel, self.channel, 1, bias=False),
                        nn.BatchNorm2d(self.channel),
                        nn.ReLU())

        self.conv_out = nn.Sequential(
                        nn.Conv2d(self.channel*2, self.channel, kernel_size=3, bias=False),
                        nn.BatchNorm2d(self.channel),
                        nn.Conv2d(self.channel, n_class, kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x_h, x_l):
        # Multi-scale features fusion.
        # x_h = self.conv_in_h(x_h)
        # x_l = self.conv_in_l(x_l)
        x_h_up = F.interpolate(x_h, size=x_l.size()[-2:], mode="bilinear", align_corners=True)
        x_l_down = F.interpolate(x_l, size=x_h.size()[-2:], mode="bilinear", align_corners=True)
        m_deep = torch.cat((x_l_down, x_h), dim=1)
        m_low = torch.cat((x_l, x_h_up), dim=1)

        # Holistic codebook generation.
        em_out = self.em(m_deep)
        base = em_out["mu"]
        x = em_out["x_trans"]

        # Codeword assembly for high-resolution feature upsampling.
        m_low = self.conv_trans_low(m_low)  # (b, 1024, h/8, w/8)
        W = self.conv_trans_attention(m_low + self.pool(x))   # (b, 1024, h/8, w/8)
        b, c, h, w = W.size()
        W = W.view(b, c, -1).permute(0, 2, 1)  # (b, h/8*w/8, 1024)
        similarity = F.softmax(torch.bmm(W, base).permute(0, 2, 1), dim=1)  # (b, k, hw)
        m_up = torch.bmm(base, similarity).view(b, c, h, w)  #(b, c, hw)
        m_up = self.rebuild_conv(m_up)

        f = torch.cat((m_up, m_low), dim=1)
        out = self.conv_out(f)

        return {"out": out,
                "base": base,
                "A": similarity.view(b, -1, h, w)}



class Detail_Branch(nn.Module):

    def __init__(self, inchannel, out_channel=128):
        super(Detail_Branch, self).__init__()

        inter_channel = out_channel//2
        self.block1 = BasicBlock(inchannel, inter_channel, stride=2)
        self.block2 = BasicBlock(inter_channel, inter_channel, stride=2)
        self.block3 = BasicBlock(inter_channel, out_channel, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x)  # 1/2
        x = self.block2(x)  # 1/4
        x = self.block3(x)  # 1/8
        return x


class Semantic_Branch(SegBaseModel):

    def __init__(self, n_class, aux=False, backbone='resnet34', pretrained_base=False, dilated=True, deep_stem=False, **kwargs):
        super(Semantic_Branch, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated, deep_stem=deep_stem, **kwargs)
        self.aux = aux
        channels = self.base_channel

        if aux:
            self.class_gcn_1 = class_gcn_2(channels[0], n_class)
            self.class_gcn_2 = class_gcn_2(channels[1], n_class)
            self.class_gcn_3 = class_gcn_2(channels[2], n_class)
            self.class_gcn_4 = class_gcn_2(channels[3], n_class)

    def forward(self, x):
        outputs = dict()
        size = x.size()[2:]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)  # 1/2  64
        x = self.backbone.maxpool(x)

        if self.aux:
            x = self.backbone.layer1(x)  # 1/4   64
            out1 = self.class_gcn_1(x)
            x = out1["out"]

            x = self.backbone.layer2(x)  # 1/8   128
            out2 = self.class_gcn_2(x)
            x = out2["out"]

            x = self.backbone.layer3(x)  # 1/16   256
            out3 = self.class_gcn_3(x)
            x = out3["out"]

            x = self.backbone.layer4(x)  # 1/32   512
            out4 = self.class_gcn_4(x)
            x = out4["out"]

            aux_outs = [out1["aux_out"], out2["aux_out"], out3["aux_out"], out4["aux_out"]]
            for i in range(len(aux_outs)):
                aux_outs[i] = F.interpolate(aux_outs[i], size=size, mode="bilinear", align_corners=True)
            outputs.update({"aux_out": aux_outs})
        else:
            x = self.backbone.layer1(x)  # 1/4   64
            x = self.backbone.layer2(x)  # 1/8   128
            x = self.backbone.layer3(x)  # 1/16   256
            x = self.backbone.layer4(x)  # 1/32   512

        outputs.update({"main_out": x})
        return outputs



class BasicBlock(nn.Module):

    def __init__(self, inchannel, channel, stride=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(channel)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out




class out_conv(nn.Module):
    def __init__(self, in_channel, n_class):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, n_class, kernel_size=1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        return self.conv(x)




















