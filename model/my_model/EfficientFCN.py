import torch
import torch.nn as nn
import torch.nn.functional as F
from model.segbase import SegBaseModel


class EfficientFCN_Module(nn.Module):
    def __init__(self, channels, n_class, n_codeword=512):
        super().__init__()
        self.n_codeword = n_codeword

        self.conv_in = nn.ModuleList()
        for channel in channels:
            self.conv_in.append(nn.Conv2d(channel, 512, kernel_size=1))

        self.conv_B = nn.Conv2d(512*3, 1024, kernel_size=1)
        self.conv_A = nn.Conv2d(512*3, n_codeword, kernel_size=1)

        self.conv_G = nn.Conv2d(512*3, 1024, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_W = nn.Conv2d(1024, n_codeword, kernel_size=1)

        self.conv_out = nn.Conv2d(1024*2, n_class, kernel_size=1)

    def forward(self, features):
        # Multi-scale features fusion.
        e = []
        for i in range(len(features)):
            e.append(self.conv_in[i](features[i]))
        down_size = e[2].size()[-2:]
        up_size = e[0].size()[-2:]
        m32 = torch.cat((F.upsample_bilinear(e[0], size=down_size), F.upsample_bilinear(e[1], size=down_size), e[2]), dim=1)
        m8 = torch.cat((e[0], F.upsample_bilinear(e[1], size=up_size), F.upsample_bilinear(e[2], size=up_size)), dim=1)

        # Holistic codebook generation.
        b, c, h, w = m32.size()
        A = F.softmax(self.conv_A(m32).reshape(b, -1, h*w), dim=-1).permute(0, 2, 1)  #weight (b, h*w, n)
        B = self.conv_B(m32)  # base code word (b, 1024, h, w)
        B_ = B.reshape(b, -1, h*w)  # (b,1024, h*w)
        code_word = torch.bmm(B_, A)  # (b, 1024, n)

        # Codeword assembly for high-resolution feature upsampling.
        G = self.conv_G(m8)  # (b, 1024, h/8, w/8)
        W = self.conv_W(G + self.pool(B))  # (b, n, h/8, w/8)
        b, c, h, w = W.size()
        W = W.view(b, c, -1)  # (b, n, h/8*w/8)
        f = torch.bmm(code_word, W).view(b, -1, h, w)

        f = torch.cat((f, G), dim=1)
        out = self.conv_out(f)

        return out




class EfficientFCN(SegBaseModel):

    def __init__(self, n_class, backbone='resnet34', aux=False, pretrained_base=False, dilated=True, deep_stem=False, **kwargs):
        super(EfficientFCN, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated, deep_stem=deep_stem, **kwargs)
        self.aux = aux
        self.dilated = dilated
        channels = self.base_channel

        self.effcient_module = EfficientFCN_Module(channels[1:], n_class)

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

        x = self.effcient_module([c3, c4, c5])
        x = F.upsample_bilinear(x, size=size)

        outputs.update({"main_out": x})

        return outputs









