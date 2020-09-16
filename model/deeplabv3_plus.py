import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_models.xception import get_xception
from .DeepLabV3 import _ASPP
from .model_utils import _FCNHead



class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class DeepLabV3Plus(nn.Module):
    r"""DeepLabV3Plus
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'xception').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """

    def __init__(self, nclass, backbone='xception', aux=False, pretrained_base=False, dilated=True, **kwargs):
        super(DeepLabV3Plus, self).__init__()
        self.aux = aux
        self.nclass = nclass
        output_stride = 8 if dilated else 32

        self.pretrained = get_xception(pretrained=pretrained_base, output_stride=output_stride, **kwargs)

        # deeplabv3 plus
        self.head = _DeepLabHead(nclass, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(728, nclass, **kwargs)

    def base_forward(self, x):
        # Entry flow
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)

        x = self.pretrained.conv2(x)
        x = self.pretrained.bn2(x)
        x = self.pretrained.relu(x)

        x = self.pretrained.block1(x)
        # add relu here
        x = self.pretrained.relu(x)
        low_level_feat = x

        x = self.pretrained.block2(x)
        x = self.pretrained.block3(x)

        # Middle flow
        x = self.pretrained.midflow(x)
        mid_level_feat = x

        # Exit flow
        x = self.pretrained.block20(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.conv3(x)
        x = self.pretrained.bn3(x)
        x = self.pretrained.relu(x)

        x = self.pretrained.conv4(x)
        x = self.pretrained.bn4(x)
        x = self.pretrained.relu(x)

        x = self.pretrained.conv5(x)
        x = self.pretrained.bn5(x)
        x = self.pretrained.relu(x)
        return low_level_feat, mid_level_feat, x

    def forward(self, x):
        size = x.size()[2:]
        c1, c3, c4 = self.base_forward(x)
        outputs = dict()
        x = self.head(c4, c1)
        # x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.update({"main_out": x})
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.update({"auxout": [auxout]})
        return outputs


class _DeepLabHead(nn.Module):
    def __init__(self, nclass, c1_channels=128, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=None, **kwargs)
        self.c1_block = _ConvBNReLU(c1_channels, 48, 3, padding=1, norm_layer=norm_layer)
        self.block = nn.Sequential(
            _ConvBNReLU(304, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, 1))

    def forward(self, x, c1):
        size = c1.size()[2:]
        c1 = self.c1_block(c1)
        x = self.aspp(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return self.block(torch.cat([x, c1], dim=1))



