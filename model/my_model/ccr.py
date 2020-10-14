import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import WeightedRandomSampler
import numpy as np


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






















