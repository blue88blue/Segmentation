from model.segbase import SegBaseModel
from model.model_utils import init_weights, _FCNHead
from .blocks import *


class EM(nn.Module):
    '''
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, c, k, stage_num=3):
        super(EM, self).__init__()
        self.stage_num = stage_num

        # 初始化基
        mu = torch.Tensor(1, c, k)  # k个描述子
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)  # 归一化
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)

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




class EMA_UP_docoder(nn.Module):
    def __init__(self, channel_h, channel_l, k=32):
        super().__init__()
        self.channel = channel_h + channel_l

        self.conv_in_h = nn.Conv2d(channel_h, channel_h, kernel_size=1)
        self.conv_in_l = nn.Conv2d(channel_l, channel_l, kernel_size=1)

        self.em = EM(self.channel, k=k)

        self.conv_trans_low = nn.Conv2d(self.channel, self.channel, kernel_size=1)

        self.conv_gobal = nn.Conv2d(self.channel, self.channel, kernel_size=1)
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
        x_h = self.conv_in_h(x_h)
        x_l = self.conv_in_l(x_l)
        x_h_up = F.interpolate(x_h, size=x_l.size()[-2:], mode="bilinear", align_corners=True)
        x_l_down = F.interpolate(x_l, size=x_h.size()[-2:], mode="bilinear", align_corners=True)
        m_deep = torch.cat((x_l_down, x_h), dim=1)
        m_low = torch.cat((x_l, x_h_up), dim=1)

        # Holistic codebook generation.
        em_out = self.em(m_deep)
        base = em_out["mu"]

        # Codeword assembly for high-resolution feature upsampling.
        m_low = self.conv_trans_low(m_low)  # (b, 1024, h/8, w/8)
        W = self.conv_trans_attention(m_low + self.pool(self.conv_gobal(m_deep)))  # (b, 1024, h/8, w/8)
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



#
# def h_position_embedding(h, device):
#     coor = []
#     for i in range(h):
#         y_range = torch.arange(-i, h-i, device=device)/float(h)
#         y_range = y_range.unsqueeze(-1).unsqueeze(-1)  # (c,1,1)
#         coor.append(y_range)
#     coor = torch.cat(coor, dim=1).unsqueeze(0).unsqueeze(0)  # (1, 1, L, h)
#     return coor
#
#
#
# class position_embedding(nn.Module):
#     def __init__(self, h, w, c, device):
#         super().__init__()
#         self.coor = h_position_embedding(h, device)
#         self.conv = nn.Conv2d(1, c, kernel_size=1)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         R = self.conv(self.coor).expand(b, -1, -1, -1).repeat(1, 1, 1, w)    # (b, c, L, h*w)
#         R = R.permute(0, 3, 1, 2).reshape(b*h*w, c, -1)
#         x_ = x.permute(0, 2, 3, 1).reshape(b*h*w, 1, c)
#         weight = torch.bmm(x_, R)  # (bhw, 1, L)
#         weight = weight.reshape(b, h, w, -1).permute(0, 3, 1, 2)  # (b, L, h, w)



def position_grid(h, w, device):
    coord_feat = []
    for i in range(h):
        y_range = torch.arange(-i, h-i, device)/float(h)
        for j in range(w):
            x_range = torch.arange(-j, w-j, device)/float(w)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.unsqueeze(0)
            x = x.unsqueeze(0)
            coord = torch.cat([x, y], 0).unsqueeze(0)
            coord_feat.append(coord)
    coord_feat = torch.cat(coord_feat, dim=0).reshape(h*w, 2, h, w)
    return coord_feat



class position_embedding(nn.Module):
    def __init__(self, h, w, c, device):
        super().__init__()
        self.coord_feat = position_grid(h, w, device)  # (hw, 2, h, w)
        self.conv = nn.Conv2d(2, c, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        R = self.conv(self.coord_feat)  # (hw, c, h, w) position_embedding











