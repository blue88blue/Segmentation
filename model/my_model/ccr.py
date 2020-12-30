import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import WeightedRandomSampler
import numpy as np
from model.segbase import SegBaseModel
from model.model_utils import init_weights, _FCNHead
from model.my_model.blocks import *
from model.my_model.SPUnet import SPSP
from .CaCNet import CaC_Module
from .TANet import Tensor_Attention
from model.my_model.Class_GCN import class_gcn_2



def softmax_T(x, dim, T=1):
    x = torch.exp(x)
    sum = torch.sum(x, dim=dim, keepdim=True)
    x = x/sum
    return x


class ccr(nn.Module):

    def __init__(self, c, k, n_class, stage_num=3):
        super(ccr, self).__init__()
        self.stage_num = stage_num
        self.k = k

        # self.conv_aux_pred = nn.Conv2d(c, n_class, kernel_size=1)

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
        b, c, h, w = x.size()
        idn = x

        # The first 1x1 conv
        x = self.conv1(x)

        # 采样
        # 深监督
        # aux_pred = self.conv_aux_pred(idn)
        # sample_weight = softmax_T(aux_pred, dim=1, T=20)
        # sample_weight = 1 - torch.max(sample_weight, dim=1)[0]  # 采样权重， 置信度越小，权重越大
        sample_weight = torch.zeros((b, h, w))  # 均匀采样
        base = []
        # base_label = []
        # label = F.interpolate(label.unsqueeze(1), size=(h, w)).squeeze(1)
        for batch in range(b):
            samples = list(WeightedRandomSampler(list(sample_weight[batch, :, :].reshape(-1)), self.k, replacement=True))
            base.append(idn.reshape(b, c, -1)[batch, :, samples].unsqueeze(0))  # (1, c, k)
            # base_label.append(label.reshape(b, -1)[batch, samples].unsqueeze(0))  # (1, k)
        base = torch.cat(base, dim=0)  # (b, c, num_sample)
        # base_label = torch.cat(base_label, dim=0)   # (b, num_sample)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = base
        for i in range(self.stage_num):
            x_t = x.permute(0, 2, 1)  # b * n * c
            z = torch.bmm(x_t, mu)  # b * n * k  # 特征图中的n个点与k个基的相似性
            z = F.softmax(z, dim=2)  # b * n * k  # 每个点属于某一个基的概率
            z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))  # 计算每个点对基的归一化的权重，
            mu = torch.bmm(x, z_)  # b * c * k  # 用每个点去加权组合，得到基
            mu = self._l2norm(mu, dim=1)

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n  # 用基重建特征图
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x

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




#
#
# class EMA_UP_docoder(nn.Module):
#     def __init__(self, channel_h, channel_l, k=32):
#         super().__init__()
#         self.channel = channel_h + channel_l
#
#         self.conv_in_h = nn.Conv2d(channel_h, channel_h, kernel_size=1)
#         self.conv_in_l = nn.Conv2d(channel_l, channel_l, kernel_size=1)
#
#         self.em = EM(self.channel, k=k)
#
#         self.conv_trans_low = nn.Conv2d(self.channel, self.channel, kernel_size=1)
#
#         self.conv_gobal = nn.Conv2d(self.channel, self.channel, kernel_size=1)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_trans_attention = nn.Conv2d(self.channel, self.channel, kernel_size=1)
#
#         self.rebuild_conv = nn.Sequential(
#                         nn.Conv2d(self.channel, self.channel, 1, bias=False),
#                         nn.BatchNorm2d(self.channel),
#                         nn.ReLU())
#         self.conv_out = nn.Sequential(
#                         nn.Conv2d(self.channel*2, channel_l, kernel_size=1, bias=False),
#                         nn.BatchNorm2d(channel_l),
#                         nn.ReLU())
#
#     def forward(self, x_h, x_l):
#         # Multi-scale features fusion.
#         x_h = self.conv_in_h(x_h)
#         x_l = self.conv_in_l(x_l)
#         x_h_up = F.interpolate(x_h, size=x_l.size()[-2:], mode="bilinear", align_corners=True)
#         x_l_down = F.interpolate(x_l, size=x_h.size()[-2:], mode="bilinear", align_corners=True)
#         m_deep = torch.cat((x_l_down, x_h), dim=1)
#         m_low = torch.cat((x_l, x_h_up), dim=1)
#
#         # Holistic codebook generation.
#         em_out = self.em(m_deep)
#         base = em_out["mu"]
#
#         # Codeword assembly for high-resolution feature upsampling.
#         m_low = self.conv_trans_low(m_low)  # (b, 1024, h/8, w/8)
#         W = self.conv_trans_attention(m_low + self.pool(self.conv_gobal(m_deep)))  # (b, 1024, h/8, w/8)
#         b, c, h, w = W.size()
#         W = W.view(b, c, -1).permute(0, 2, 1)  # (b, h/8*w/8, 1024)
#         similarity = F.softmax(torch.bmm(W, base).permute(0, 2, 1), dim=1)  # (b, k, hw)
#         m_up = torch.bmm(base, similarity).view(b, c, h, w)  #(b, c, hw)
#         m_up = self.rebuild_conv(m_up)
#
#         f = torch.cat((m_up, m_low), dim=1)
#         out = self.conv_out(f)
#
#         return {"out": out,
#                 "base": base,
#                 "A": similarity.view(b, -1, h, w)}




class EMA_UP_docoder_EM_x(nn.Module):
    def __init__(self, channel_h, channel_l, k=64):
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
                        nn.Conv2d(self.channel*2, channel_l, kernel_size=1, bias=False),
                        nn.BatchNorm2d(channel_l),
                        nn.ReLU())

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



#  残差连接
class EMA_UP_docoder(nn.Module):
    def __init__(self, channel_h, channel_l, k=64):
        super().__init__()
        self.channel = channel_h + channel_l

        self.conv_in_h = conv_bn_relu(channel_h, channel_h, kernel_size=1, padding=0)
        self.conv_in_l = conv_bn_relu(channel_l, channel_l, kernel_size=1, padding=0)

        self.em = EM(self.channel, k=k)

        self.conv_trans_low = nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_trans_attention = nn.Conv2d(self.channel, self.channel, kernel_size=1)

        self.rebuild_conv = nn.Sequential(
                        nn.Conv2d(self.channel, self.channel, 1, bias=False),
                        nn.BatchNorm2d(self.channel))

        self.conv_out = nn.Sequential(
            nn.Conv2d(self.channel, channel_l, 1, bias=False),
            nn.BatchNorm2d(channel_l),
            nn.ReLU())

    def forward(self, x_h, x_l):

        # Multi-scale features fusion.
        x_h = self.conv_in_h(x_h)
        x_l = self.conv_in_l(x_l)
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

        # fusion
        out = self.conv_out(m_up + m_low)

        return {"out": out,
                "base": base,
                "A": similarity.view(b, -1, h, w)}







class EM(nn.Module):
    '''
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, c, k, stage_num=3, inter_channel=None):
        super(EM, self).__init__()
        self.stage_num = stage_num
        if inter_channel == None:
            inter_channel = c

        # 初始化基
        mu = torch.Tensor(1, inter_channel, k)  # k个描述子
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)  # 归一化
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, inter_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # The first 1x1 conv
        x = self.conv1(x)
        x_trans = x
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
        return {"mu": mu,
                "x_trans": x_trans}

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



class Attention_UP_decoder(nn.Module):
    def __init__(self,  channel_h, channel_l, n_codeword=512):
        super().__init__()
        self.n_codeword = n_codeword
        self.channel = channel_h + channel_l

        self.conv_in_h = conv_bn_relu(channel_h, channel_h, kernel_size=1, padding=0)
        self.conv_in_l = conv_bn_relu(channel_l, channel_l, kernel_size=1, padding=0)

        self.conv_B = nn.Conv2d(self.channel, self.channel, kernel_size=1)
        self.conv_A = nn.Conv2d(self.channel, n_codeword, kernel_size=1)

        self.conv_G = nn.Conv2d(self.channel, self.channel, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_W = nn.Conv2d(self.channel, n_codeword, kernel_size=1)

        self.conv_out = nn.Conv2d(self.channel*2, channel_l, kernel_size=1)

    def forward(self, x_h, x_l):
        # Multi-scale features fusion.
        x_h = self.conv_in_h(x_h)
        x_l = self.conv_in_l(x_l)
        x_h_up = F.interpolate(x_h, size=x_l.size()[-2:], mode="bilinear", align_corners=True)
        x_l_down = F.interpolate(x_l, size=x_h.size()[-2:], mode="bilinear", align_corners=True)
        m_deep = torch.cat((x_l_down, x_h), dim=1)
        m_low = torch.cat((x_l, x_h_up), dim=1)

        # Holistic codebook generation.
        b, c, h, w = m_deep.size()
        A = F.softmax(self.conv_A(m_deep).reshape(b, -1, h*w), dim=-1).permute(0, 2, 1)  #weight (b, h*w, n)
        B = self.conv_B(m_deep)  # base code word (b, 1024, h, w)
        B_ = B.reshape(b, -1, h*w)  # (b,1024, h*w)
        code_word = torch.bmm(B_, A)  # (b, 1024, n)

        # Codeword assembly for high-resolution feature upsampling.
        G = self.conv_G(m_low)  # (b, 1024, h/8, w/8)
        W = self.conv_W(G + self.pool(B))  # (b, n, h/8, w/8)
        b, c, h, w = W.size()
        W = W.view(b, c, -1)  # (b, n, h/8*w/8)
        f = torch.bmm(code_word, W).view(b, -1, h, w)

        f = torch.cat((f, G), dim=1)
        out = self.conv_out(f)

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




class CMSI(nn.Module):
    def __init__(self, in_channel, scales):
        super().__init__()
        assert in_channel % 4 == 0
        out_channel = int(in_channel / 4)

        self.conv1 = conv_bn_relu(in_channel, out_channel)
        self.conv2 = conv_bn_relu(out_channel, out_channel)
        self.conv3 = conv_bn_relu(out_channel, out_channel)
        self.conv4 = conv_bn_relu(out_channel, out_channel, kernel_size=1, padding=0)

        self.conv2_1 = conv_bn_relu(in_channel+out_channel, out_channel)
        self.conv2_2 = conv_bn_relu(out_channel*2, out_channel)
        self.conv2_3 = conv_bn_relu(out_channel*2, out_channel, kernel_size=1, padding=0)

        self.conv3_1 = conv_bn_relu(in_channel+out_channel, out_channel)
        self.conv3_2 = conv_bn_relu(out_channel * 2, out_channel, kernel_size=1, padding=0)

        self.conv4_1 = conv_bn_relu(in_channel + out_channel, out_channel, kernel_size=1, padding=0)

        self.pool1 = nn.AdaptiveAvgPool2d((scales[0], scales[0]))
        self.pool2 = nn.AdaptiveAvgPool2d((scales[1], scales[1]))
        self.pool3 = nn.AdaptiveAvgPool2d((scales[2], scales[2]))
        self.pool4 = nn.AdaptiveAvgPool2d((scales[3], scales[3]))

        self.sconv1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, (1, 3), 1, (0, 1), bias=False),
                                    nn.BatchNorm2d(out_channel))
        self.sconv2 = nn.Sequential(nn.Conv2d(in_channel, out_channel, (3, 1), 1, (1, 0), bias=False),
                                    nn.BatchNorm2d(out_channel))
        self.spool1 = nn.AdaptiveAvgPool2d((1, None))
        self.spool2 = nn.AdaptiveAvgPool2d((None, 1))

        self.conv_out = conv_bn_relu(in_channel*2+out_channel, in_channel)

    def forward(self, x):
        b, c, h, w = x.size()

        x1 = self.conv1(self.pool1(x))
        x2 = self.conv2(self.pool2(x1))
        x3 = self.conv3(self.pool3(x2))
        x4 = self.conv4(self.pool4(x3))

        x2_1 = self.conv2_1(torch.cat((self.pool2(x), x2), dim=1))
        x2_2 = self.conv2_2(torch.cat((self.pool3(x2_1), x3), dim=1))
        x2_3 = self.conv2_3(torch.cat((self.pool4(x2_2), x4), dim=1))

        x3_1 = self.conv3_1(torch.cat((self.pool3(x), x2_2), dim=1))
        x3_2 = self.conv3_2(torch.cat((self.pool4(x3_1), x2_3), dim=1))

        x4_1 = self.conv4_1(torch.cat((self.pool4(x), x3_2), dim=1))

        # 上采样
        y1 = F.interpolate(x1, size=(h, w), mode="bilinear", align_corners=True)
        y2 = F.interpolate(x2_1, size=(h, w), mode="bilinear", align_corners=True)
        y3 = F.interpolate(x3_1, size=(h, w), mode="bilinear", align_corners=True)
        y4 = F.interpolate(x4_1, size=(h, w), mode="bilinear", align_corners=True)

        # 条形池化
        x5 = F.interpolate(self.sconv1(self.spool1(x)), size=(h, w), mode="bilinear", align_corners=True)
        x6 = F.interpolate(self.sconv2(self.spool2(x)), size=(h, w), mode="bilinear", align_corners=True)
        y5 = F.relu(x5 + x6)

        # concat
        out = torch.cat((x, y1, y2, y3, y4, y5), dim=1)
        out = self.conv_out(out)

        return out











class EMUPNet(SegBaseModel):

    def __init__(self,  n_class, image_size=None,  backbone='resnet34', pretrained_base=False, deep_stem=False, **kwargs):
        super(EMUPNet, self).__init__(backbone, pretrained_base=pretrained_base, deep_stem=deep_stem, **kwargs)
        channels = self.base_channel  # [256, 512, 1024, 2048]
        if deep_stem or backbone == 'resnest101':
            conv1_channel = 128
        else:
            conv1_channel = 64

        # self.class_gcn = class_gcn_2(channels[3], n_class)

        self.donv_up1 = EMA_UP_docoder(channels[3], channels[2], k=64)
        self.donv_up2 = EMA_UP_docoder(channels[2], channels[1], k=64)
        self.donv_up3 = EMA_UP_docoder(channels[1], channels[0], k=64)
        self.donv_up4 = EMA_UP_docoder(channels[0], conv1_channel, k=64)

        self.out_conv = out_conv(conv1_channel, n_class)


    def forward(self, x):
        outputs = dict()
        size = x.size()[2:]

        c1, c2, c3, c4, c5 = self.backbone.extract_features(x)

        # aux_out, c5 = self.class_gcn(c5)
        # aux_out = F.interpolate(aux_out, size, mode='bilinear', align_corners=True)
        # outputs.update({"aux_out": [aux_out]})

        x1 = self.donv_up1(c5, c4)

        x2 = self.donv_up2(x1["out"], c3)

        x3 = self.donv_up3(x2["out"], c2)

        x4 = self.donv_up4(x3["out"], c1)

        x = self.out_conv(x4["out"])
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 最后上采样

        outputs.update({"main_out": x})
        outputs.update({"mu1": x1["base"],
                        "mu2": x2["base"],
                        "mu3": x3["base"],
                        "mu4": x4["base"]})
        outputs.update({"A1": x1["A"],
                        "A2": x2["A"],
                        "A3": x3["A"],
                        "A4": x4["A"]})
        return outputs

#
# class EMUPNet(SegBaseModel):
#
#     def __init__(self, n_class, image_size=None, backbone='resnet34', pretrained_base=False, deep_stem=False, **kwargs):
#         super(EMUPNet, self).__init__(backbone, pretrained_base=pretrained_base, deep_stem=deep_stem, **kwargs)
#         channels = self.base_channel  # [256, 512, 1024, 2048]
#         if deep_stem or backbone == 'resnest101':
#             conv1_channel = 128
#         else:
#             conv1_channel = 64
#
#         self.donv_up1 = Attention_UP_decoder(channels[3], channels[2])
#         self.donv_up2 = Attention_UP_decoder(channels[2], channels[1])
#         self.donv_up3 = Attention_UP_decoder(channels[1], channels[0])
#         self.donv_up4 = Attention_UP_decoder(channels[0], conv1_channel)
#
#         self.out_conv = out_conv(conv1_channel, n_class)
#
#
#     def forward(self, x):
#         outputs = dict()
#         size = x.size()[2:]
#
#         c1, c2, c3, c4, c5 = self.backbone.extract_features(x)
#
#         x = self.donv_up1(c5, c4)
#         x = self.donv_up2(x, c3)
#         x = self.donv_up3(x, c2)
#         x = self.donv_up4(x, c1)
#
#         x = self.out_conv(x)
#         x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 最后上采样
#
#         outputs.update({"main_out": x})
#
#         return outputs




