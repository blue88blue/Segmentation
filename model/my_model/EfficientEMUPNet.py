import torch
import torch.nn as nn
import torch.nn.functional as F
from model.segbase import SegBaseModel
from model.my_model.blocks import *
from model.my_model.ccr import EM

class EfficientEMUP_Module(nn.Module):
    def __init__(self, channels, n_class, n_codeword=64, inter_channel=512):
        super().__init__()
        self.n_codeword = n_codeword
        self.sum_channel = sum(channels)

        self.conv_in = nn.ModuleList()
        for channel in channels:
            self.conv_in.append(conv_bn_relu(channel, channel, kernel_size=1, padding=0))


        self.em = EM(self.sum_channel, k=n_codeword, inter_channel=inter_channel)

        self.conv_trans_low = nn.Conv2d(self.sum_channel, inter_channel, kernel_size=1, padding=0)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_trans_attention = nn.Conv2d(inter_channel, inter_channel, kernel_size=1)

        self.rebuild_conv = nn.Sequential(
                        nn.Conv2d(inter_channel, inter_channel, 1, bias=False),
                        nn.BatchNorm2d(inter_channel),
                        nn.ReLU())
        self.conv_out = nn.Conv2d(inter_channel*2, n_class, kernel_size=1, bias=False)

    def forward(self, features):
        # Multi-scale features fusion.
        e = []
        for i in range(len(features)):
            e.append(self.conv_in[i](features[i]))
        down_size = e[3].size()[-2:]
        up_size = e[0].size()[-2:]
        m_low = torch.cat((F.interpolate(e[0], size=down_size, mode="bilinear", align_corners=True),
                        F.interpolate(e[1], size=down_size, mode="bilinear", align_corners=True),
                         F.interpolate(e[2], size=down_size, mode="bilinear", align_corners=True), e[3]), dim=1)
        m_deep = torch.cat((e[0], F.interpolate(e[1], size=up_size, mode="bilinear", align_corners=True),
                        F.interpolate(e[2], size=up_size, mode="bilinear", align_corners=True),
                            F.interpolate(e[3], size=up_size, mode="bilinear", align_corners=True)), dim=1)

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
                "base": base}






class EfficientEMUPNet(SegBaseModel):

    def __init__(self, n_class, backbone='resnet34', aux=False, pretrained_base=False, dilated=False, deep_stem=False, **kwargs):
        super(EfficientEMUPNet, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated, deep_stem=deep_stem, **kwargs)
        self.aux = aux
        self.dilated = dilated
        channels = self.base_channel

        self.effcient_module = EfficientEMUP_Module(channels, n_class)

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

        out = self.effcient_module([c2, c3, c4, c5])
        x = out["out"]
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)

        outputs.update({"main_out": x})
        outputs.update({"mu": out["base"]})
        return outputs