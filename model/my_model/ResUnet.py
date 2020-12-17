from model.segbase import SegBaseModel
from model.model_utils import init_weights, _FCNHead
from .blocks import *
from .SPUnet import SPSP
from .ccr import ccr


class ResUnet(SegBaseModel):

    def __init__(self, n_class, backbone='resnet34', aux=False, pretrained_base=False, dilated=True, deep_stem=False, **kwargs):
        super(ResUnet, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated, deep_stem=deep_stem, **kwargs)
        self.aux = aux
        self.dilated = dilated
        channels = self.base_channel
        if deep_stem or backbone == 'resnest101':
            conv1_channel = 128
        else:
            conv1_channel = 64

        if dilated:
            self.donv_up3 = decoder_block(channels[0]+channels[3], channels[0])
            self.donv_up4 = decoder_block(channels[0]+conv1_channel, channels[0])
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

        if self.dilated:
            x = self.donv_up3(c5, c2)
            x = self.donv_up4(x, c1)
        else:
            x = self.donv_up1(c5, c4)
            x = self.donv_up2(x, c3)
            x = self.donv_up3(x, c2)
            x = self.donv_up4(x, c1)

        outputs.update({"feature": x})

        x = self.out_conv(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 最后上采样

        outputs.update({"main_out": x})
        if self.aux:
            auxout = self.aux_layer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.update({"aux_out": [auxout]})
        return outputs




