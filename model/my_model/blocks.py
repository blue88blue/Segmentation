import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)

        return x

class SeparableConv2d_BN_RELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d_BN_RELU, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.BN(x)
        x = self.relu(x)
        return x



class decoder_block(nn.Module):
    def __init__(self, in_channel, out_channel, se=False):
        super().__init__()

        self.dconv = nn.Sequential(

            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

            # nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channel),
            # nn.ReLU(),
        )

        self.se = se
        if se:
            self.selayer = ECA_layer(out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x_h, x_l):
        # x_h = F.interpolate(x_h, size=x_l.size()[-2:], mode="bilinear", align_corners=True)
        x = torch.cat((x_h, x_l), dim=1)
        x = self.dconv(x)
        if self.se:
            x = self.selayer(x)
        return x






class conv_bn_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, dilation=1, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)




class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias






class double_conv(nn.Module):
    def __init__(self, in_channel, out_channel, se=False):
        super().__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.se = se
        if se:
            self.selayer = ECA_layer(out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.dconv(x)
        if self.se:
            x = self.selayer(x)
        return x




class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class ECA_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, gamma=2, b=1):
        super(ECA_layer, self).__init__()
        t = int(abs(math.log(channel, 2) + b) / gamma)
        k = t if t%2 else t+1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  # (b, c, 1, 1)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)






class ECA_fusion_layer(nn.Module):
    '''
    将多个特征图的通道信息融合， 得到每个特征图的通道注意力
    '''
    def __init__(self, channel, fusion_layers,  gamma=2, b=1):
        super(ECA_fusion_layer, self).__init__()
        t = int(abs(math.log(channel, 2) + b) / gamma)
        k = t if t%2 else t+1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(fusion_layers, fusion_layers, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is a list of tensor
        # feature descriptor on the global spatial information
        y = []
        for i in range(len(x)):
            y.append(self.avg_pool(x[i]))  # (b, c, 1, 1)
        y = torch.cat(y, dim=2)   # (b, c, len(x), 1)

        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        for i in range(len(x)):
            y_i = y[:, :, i, :].unsqueeze(2)
            x[i] = x[i] * y_i.expand_as(x[i])

        return x * y.expand_as(x)





class non_local(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)


    def forward(self, x):


        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # N*(WH)*C

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # N*(WH)*C

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # N*C*(WH)
        f = torch.matmul(theta_x, phi_x)  # HW*HW
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)  # N*(WH)*C
        y = y.permute(0, 2, 1).contiguous()  # N*C*(WH)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # N*C*H*W

        W_y = self.W(y)
        z = W_y + x

        return z





# class Adaptive_Context_Block(nn.Module):
#     '''
#     只用全局相似度
#     '''
#     def __init__(self, channel_h, channel_l, out_channel):
#         super().__init__()
#
#         self.global_ = global_guide_gate(channel_h)
#         self.alpha = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=True)
#
#         self.conv = decoder_block(channel_h + channel_l, out_channel)
#
#     def forward(self, x_h, x_l):
#         size_l = x_l.size()[-2:]  # low level
#         # 计算全局和局部门控
#         global_gate, global_feature = self.global_(x_h)  # (b, 1, h, w),  (b, c, h, w)
#         global_gate_up = F.interpolate(global_gate, size=size_l, mode="bilinear", align_corners=True)  # 上采样
#
#         # 自适应上下文融合
#         y_l = (1-global_gate_up) * x_l
#         y_h = (self.alpha * global_gate) * global_feature + x_h
#
#         # 高低语义特征图concat合并+卷积
#         y_h = F.interpolate(y_h, size=size_l, mode="bilinear", align_corners=True)  # 上采样
#         y = torch.cat((y_h, y_l), dim=1)
#         y = self.conv(y)
#
#         return y





class Attention_Gate(nn.Module):
    def __init__(self, in_channel_x, in_channel_g, channel_reduce=2):
        super(Attention_Gate, self).__init__()
        self.in_channel_x = in_channel_x
        self.in_channel_g = in_channel_g
        self.inter_channel = max(in_channel_x//channel_reduce, 16)

        self.Wx = nn.Sequential(
            nn.Conv2d(in_channel_x, self.inter_channel, kernel_size=1),
            nn.BatchNorm2d(self.inter_channel)
        )
        self.Wg = nn.Sequential(
            nn.Conv2d(in_channel_g, self.inter_channel, kernel_size=1),
            nn.BatchNorm2d(self.inter_channel)
        )
        self.phi =nn.Sequential(
            nn.Conv2d(self.inter_channel, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        size = x.size()[-2:]
        g_ = F.interpolate(self.Wg(g), size=size, mode="bilinear", align_corners=True)
        x_ = self.Wx(x)

        f = self.phi(F.relu(g_ + x_))

        x_out = f.expand_as(x) * x

        return x_out



# class group_pyramid_conv(nn.Module):
#     def __init__(self, in_channel, scales):
#         super().__init__()
#         assert in_channel % 4 == 0
#         self.group_channel = in_channel // 4
#         self.conv_split = conv_bn_relu(in_channel, in_channel, kernel_size=1, padding=0)
#
#         self.conv_in = conv_bn_relu(in_channel, self.group_channel)
#         self.conv1 = conv_bn_relu(self.group_channel*2, self.group_channel)
#         self.conv2 = conv_bn_relu(self.group_channel*2, self.group_channel)
#         self.conv3 = conv_bn_relu(self.group_channel*2, self.group_channel)
#         self.conv4 = conv_bn_relu(self.group_channel*2, self.group_channel)
#         self.conv_out = conv_bn_relu(self.group_channel*5, in_channel, kernel_size=1, padding=0)
#
#         self.pool1 = nn.AdaptiveAvgPool2d((scales[0], scales[0]))
#         self.pool2 = nn.AdaptiveAvgPool2d((scales[1], scales[1]))
#         self.pool3 = nn.AdaptiveAvgPool2d((scales[2], scales[2]))
#         self.pool4 = nn.AdaptiveAvgPool2d((scales[3], scales[3]))
#
#     def forward(self, x):
#         idn = x
#         b, c, h, w = x.size()
#         # 分组
#         x = self.conv_split(x)
#         x1 = x[:, 0:self.group_channel, :, :]
#         x2 = x[:, self.group_channel:2*self.group_channel, :, :]
#         x3 = x[:, 2*self.group_channel:3*self.group_channel, :, :]
#         x4 = x[:, 3*self.group_channel:, :, :]
#         # 级联卷积
#         y = self.conv_in(x)
#         y1 = self.conv1(torch.cat((self.pool1(y), self.pool1(x1)), dim=1))
#         y2 = self.conv2(torch.cat((self.pool2(y1), self.pool2(x2)), dim=1))
#         y3 = self.conv3(torch.cat((self.pool3(y2), self.pool3(x3)), dim=1))
#         y4 = self.conv4(torch.cat((self.pool4(y3), self.pool4(x4)), dim=1))
#         # 上采样
#         y1 = F.interpolate(y1, size=(h, w), mode="bilinear", align_corners=True)
#         y2 = F.interpolate(y2, size=(h, w), mode="bilinear", align_corners=True)
#         y3 = F.interpolate(y3, size=(h, w), mode="bilinear", align_corners=True)
#         y4 = F.interpolate(y4, size=(h, w), mode="bilinear", align_corners=True)
#         # concat
#         out = torch.cat((y, y1, y2, y3, y4), dim=1)
#         out = self.conv_out(out) + idn
#
#         return out


class group_pyramid_conv(nn.Module):
    def __init__(self, in_channel, dilations):
        super().__init__()
        assert in_channel % 4 == 0
        self.group_channel = in_channel // 4
        # self.conv_split = conv_bn_relu(in_channel, in_channel, kernel_size=1, padding=0)

        self.aux_conv = conv_bn_relu(in_channel, self.group_channel*2)

        self.conv_in = conv_bn_relu(in_channel, self.group_channel)
        self.conv1 = conv_bn_relu(self.group_channel, self.group_channel, dilation=dilations[0], padding=dilations[0])
        self.conv2 = conv_bn_relu(self.group_channel, self.group_channel, dilation=dilations[1], padding=dilations[1])
        self.conv3 = conv_bn_relu(self.group_channel, self.group_channel, dilation=dilations[2], padding=dilations[2])
        self.conv4 = conv_bn_relu(self.group_channel, self.group_channel, dilation=dilations[3], padding=dilations[3])

        self.conv_out = nn.Sequential(
            nn.Conv2d(self.group_channel*7, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        idn = self.aux_conv(x)
        # 分组
        # x = self.conv_split(x)
        x1 = x[:, 0:self.group_channel, :, :]
        x2 = x[:, self.group_channel:2*self.group_channel, :, :]
        x3 = x[:, 2*self.group_channel:3*self.group_channel, :, :]
        x4 = x[:, 3*self.group_channel:, :, :]
        # 级联卷积
        y = self.conv_in(x)
        y1 = self.conv1(y+x1)
        y2 = self.conv2(y1+x2)
        y3 = self.conv3(y2+x3)
        y4 = self.conv4(y3+x4)

        # concat
        out = torch.cat((idn, y, y1, y2, y3, y4), dim=1)
        out = self.conv_out(out)

        return out




def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class group_pyramid_conv_V2(nn.Module):
    def __init__(self, in_channel, dilations):
        super().__init__()
        assert in_channel % 4 == 0
        self.group_channel = in_channel // 4

        self.banch2 = nn.Sequential(
            # pw
            nn.Conv2d(self.group_channel, self.group_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.group_channel),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.group_channel, self.group_channel, 3, padding=dilations[0], dilation=dilations[0],
                      bias=False),
            nn.BatchNorm2d(self.group_channel),
            # pw-linear
            nn.Conv2d(self.group_channel, self.group_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.group_channel),
            nn.ReLU(inplace=True),
        )
        self.banch3 = nn.Sequential(
            # pw
            nn.Conv2d(self.group_channel*2, self.group_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.group_channel),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.group_channel, self.group_channel, 3, padding=dilations[1], dilation=dilations[1],
                      bias=False),
            nn.BatchNorm2d(self.group_channel),
            # pw-linear
            nn.Conv2d(self.group_channel, self.group_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.group_channel),
            nn.ReLU(inplace=True),
        )
        self.banch4 = nn.Sequential(
            # pw
            nn.Conv2d(self.group_channel*2, self.group_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.group_channel),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.group_channel, self.group_channel, 3, padding=dilations[2], dilation=dilations[2],
                      bias=False),
            nn.BatchNorm2d(self.group_channel),
            # pw-linear
            nn.Conv2d(self.group_channel, self.group_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.group_channel),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        # 分组
        x1 = x[:, 0:self.group_channel, :, :]
        x2 = x[:, self.group_channel:2*self.group_channel, :, :]
        x3 = x[:, 2*self.group_channel:3*self.group_channel, :, :]
        x4 = x[:, 3*self.group_channel:, :, :]
        # 级联卷积
        y2 = self.banch2(x2)
        y3 = self.banch3(torch.cat((y2, x3), dim=1))
        y4 = self.banch4(torch.cat((y3, x4), dim=1))

        # concat
        out = torch.cat((x1, y2, y3, y4), dim=1)
        return channel_shuffle(out, 4)







class class_attention(nn.Module):
    def __init__(self, channel, n_class):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.enc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.ReLU(inplace=True),
        )

        self.fc_class = nn.Sequential(
            nn.Linear(channel, n_class, bias=False),
        )

        self.fc_atention = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        enc = self.enc(y)

        class_pred = self.fc_class(enc)
        # attention = self.fc_atention(enc).view(b, c, 1, 1)

        return class_pred





class GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2
        # kernel size had better be odd number so as to avoid alignment error
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x






class AlignModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(AlignModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(outplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature= x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature,size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output



