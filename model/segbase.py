"""Base Model for Semantic Segmentation"""
import torch.nn as nn

from .base_models.resnetv1b import resnet50_v1b, resnet101_v1b, resnet152_v1b, resnet34_v1b
from .base_models.densenet import *
from .base_models.EfficientNet.model import EfficientNet
from .base_models.resnext import resnext34
from .base_models.resnest import resnest50, resnest101

class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, backbone='resnet50', pretrained_base=False, dilated=False, **kwargs):
        super(SegBaseModel, self).__init__()

        if backbone == "resnet34":
            self.backbone = resnet34_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.base_channel = [64, 128, 256, 512]
        elif backbone == "resnext34":
            self.backbone = resnext34(dilated=dilated, **kwargs)
            self.base_channel = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.backbone = resnet50_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            self.backbone = resnet101_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnet152':
            self.backbone = resnet152_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest50':
            self.backbone = resnest50(pretrained=pretrained_base, dilated=dilated)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest101':
            self.backbone = resnest101(pretrained=pretrained_base, dilated=dilated)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == "densenet121":
            if dilated:
                self.backbone = dilated_densenet121(8, pretrained=pretrained_base)
            else:
                self.backbone = densenet121(pretrained=pretrained_base)
            self.base_channel = self.backbone.num_features
        elif backbone == "densenet169":
            if dilated:
                self.backbone = dilated_densenet169(8, pretrained=pretrained_base)
            else:
                self.backbone = densenet169(pretrained=pretrained_base)
            self.base_channel = self.backbone.num_features
        elif backbone == "densenet201":
            if dilated:
                self.backbone = dilated_densenet201(8, pretrained=pretrained_base)
            else:
                self.backbone = densenet201(pretrained=pretrained_base)
            self.base_channel = self.backbone.num_features
        elif "efficientnet" in backbone:
            if pretrained_base:
                self.backbone = EfficientNet.from_pretrained(backbone)
            else:
                self.backbone = EfficientNet.from_name(backbone)
            # self.base_channel = [24, 48, 120, 352]  # b2 16
            self.base_channel = [32, 56, 160, 448]  # b4  24
            self.base_channel = [32, 48, 136, 384]  # b3 24
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        # 冻结参数
        # for p in self.backbone.conv1.parameters():
        #     p.requires_grad = False
        # for p in self.backbone.bn1.parameters():
        #     p.requires_grad = False
        # for p in self.backbone.layer1.parameters():
        #     p.requires_grad = False
        # for p in self.backbone.layer2.parameters():
        #     p.requires_grad = False


    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)

        return c1, c2, c3, c4

