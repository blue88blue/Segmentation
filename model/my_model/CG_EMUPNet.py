from model.segbase import SegBaseModel
from model.model_utils import init_weights, _FCNHead
from model.my_model.blocks import *
from model.my_model.ccr import EMA_UP_docoder, out_conv
from model.my_model.Class_GCN import class_gcn_2





class CG_EMUPNet(SegBaseModel):

    def __init__(self,  n_class, image_size=None,  backbone='resnet34', pretrained_base=False, deep_stem=False, **kwargs):
        super(CG_EMUPNet, self).__init__(backbone, pretrained_base=pretrained_base, deep_stem=deep_stem, **kwargs)
        channels = self.base_channel  # [256, 512, 1024, 2048]
        if deep_stem or backbone == 'resnest101':
            conv1_channel = 128
        else:
            conv1_channel = 64

        self.class_gcn1 = class_gcn_2(channels[3], n_class)
        self.class_gcn2 = class_gcn_2(channels[2], n_class)
        self.class_gcn3 = class_gcn_2(channels[1], n_class)
        self.class_gcn4 = class_gcn_2(channels[0], n_class)

        self.donv_up1 = EMA_UP_docoder(channels[3], channels[2], k=64)
        self.donv_up2 = EMA_UP_docoder(channels[2], channels[1], k=64)
        self.donv_up3 = EMA_UP_docoder(channels[1], channels[0], k=64)
        self.donv_up4 = EMA_UP_docoder(channels[0], conv1_channel, k=64)

        self.out_conv = out_conv(conv1_channel, n_class)


    def forward(self, x):
        outputs = dict()
        size = x.size()[2:]

        c1, c2, c3, c4, c5 = self.backbone.extract_features(x)

        out_gcn = self.class_gcn1(c5)
        c5 = out_gcn["out"]
        aux_pred = out_gcn["aux_pred"]
        aux_out = []
        aux_out.append(F.interpolate(out_gcn["aux_out"], size, mode='bilinear', align_corners=True))
        outputs.update({"aux_out": aux_out})

        c4 = self.class_gcn2(c4, F.interpolate(aux_pred, c4.size()[-2:], mode='bilinear', align_corners=True))["out"]
        c3 = self.class_gcn3(c3, F.interpolate(aux_pred, c3.size()[-2:], mode='bilinear', align_corners=True))["out"]
        c2 = self.class_gcn4(c2, F.interpolate(aux_pred, c2.size()[-2:], mode='bilinear', align_corners=True))["out"]

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


















