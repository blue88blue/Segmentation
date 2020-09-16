import torch
import torch.nn as nn
import torch.nn.functional as F


class Tensor_Generation_Reconstruction_Module(nn.Module):
    def __init__(self, c, h, w, r):
        super().__init__()
        self.c = c
        self.h = h
        self.w = w
        self.r = r
        self.conv_channel = nn.Sequential(
            nn.Conv2d(c, c*r, kernel_size=1, bias=False),
            nn.BatchNorm2d(c*r)
        )
        self.conv_hight = nn.Sequential(
            nn.Conv2d(h, h*r, kernel_size=1, bias=False),
            nn.BatchNorm2d(h*r)
        )
        self.conv_width = nn.Sequential(
            nn.Conv2d(w, w*r, kernel_size=1, bias=False),
            nn.BatchNorm2d(w*r)
        )
        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        self.lamda = nn.Parameter(torch.ones((r), dtype=torch.float32)/8, requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # tensor生成
        x_channel = self.ave_pool(x)
        x_hight = self.ave_pool(x.permute(0, 2, 1, 3))
        x_width = self.ave_pool(x.permute(0, 3, 1, 2))

        x_channel = self.sigmoid(self.conv_channel(x_channel)).squeeze()  # (b, c*r)
        x_hight = self.sigmoid(self.conv_hight(x_hight)).squeeze()  # (b, h*r)
        x_width = self.sigmoid(self.conv_width(x_width)).squeeze()  # (b, w*r)

        x_channel = torch.split(x_channel, self.c, dim=1)
        x_hight = torch.split(x_hight, self.h, dim=1)
        x_width = torch.split(x_width, self.w, dim=1)

        # tensor重建
        A = 0
        for i in range(self.r):
            v_c = x_channel[i].unsqueeze(-1).unsqueeze(-1)  # # (b, c, 1, 1)
            v_h = x_hight[i].unsqueeze(1).unsqueeze(-1)  # # (b, 1, h, 1)
            v_w = x_width[i].unsqueeze(1).unsqueeze(1)  # # (b, 1, 1, w)
            A += (v_c * v_h * v_w) * self.lamda[i]
        x = A * x

        return x


class Reco_module(nn.Module):
    def __init__(self, c, h, w, r):
        super().__init__()
        self.TRGM = Tensor_Generation_Reconstruction_Module(c, h, w, r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(True),
        )
    def forward(self, x):
        new_x = self.TRGM(x)
        global_x = self.conv(self.pool(x))
        global_x = F.interpolate(global_x, size=x.size()[-2:], mode="bilinear", align_corners=True)

        out = torch.cat((new_x, x, global_x), dim=1)
        return out



















