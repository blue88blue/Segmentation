import torch
import torch.nn as nn
import torch.nn.functional as F
from model.segbase import SegBaseModel
from ..model_utils import init_weights, _FCNHead
from .blocks import double_conv, decoder_block, conv_bn_relu


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



class CMSINet(SegBaseModel):

    def __init__(self, n_class, backbone='resnet34', aux=False, pretrained_base=False, dilated=True,
                 deep_stem=False, **kwargs):
        super(CMSINet, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated,
                                      deep_stem=deep_stem, **kwargs)
        self.aux = aux
        self.dilated = dilated
        channels = self.base_channel
        if deep_stem or backbone == 'resnest101':
            conv1_channel = 128
        else:
            conv1_channel = 64

        self.spsp = CMSI(channels[3], scales=[7, 3, 2, 1])

        if dilated:
            self.donv_up3 = decoder_block(channels[0] + channels[3], channels[0])
            self.donv_up4 = decoder_block(channels[0] + conv1_channel, channels[0])
        else:
            self.donv_up1 = decoder_block(channels[2] + channels[3], channels[2])
            self.donv_up2 = decoder_block(channels[1] + channels[2], channels[1])
            self.donv_up3 = decoder_block(channels[0] + channels[1], channels[0])
            self.donv_up4 = decoder_block(channels[0] + conv1_channel, channels[0])

        if self.aux:
            self.aux_layer = _FCNHead(256, n_class)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
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

        c5 = self.spsp(c5)

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
        if self.aux:
            auxout = self.aux_layer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.update({"auxout": [auxout]})
        return outputs