from model.segbase import SegBaseModel
from model.model_utils import init_weights, _FCNHead
from .blocks import *


class base_generater(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.conv_key = nn.Conv2d(c, c, kernel_size=1)
        self.conv_query = nn.Conv2d(c, k, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        key = self.conv_key(x).view(b, c, h*w).contiguous()  # (b, c, hw)
        query = self.conv_query(x).view(b, -1, h*w).permute(0, 2, 1).contiguous()  # (b, hw, k)
        query = F.softmax(query, dim=1)

        feature = torch.bmm(key, query).unsqueeze(-1)  # (b, c, k, 1)
        feature = self.conv(feature).squeeze(-1)

        return feature


class Tensor_Attention(nn.Module):
    def __init__(self, h, w, c, k=64):
        super().__init__()
        self.k = k
        self.G_channel = base_generater(c, k=k)
        self.G_high = base_generater(h, k=k)
        self.G_width = base_generater(w, k=k)

        # self.conv_out = nn.Sequential(
        #     nn.Conv2d(c*2, c, kernel_size=1),
        #     nn.BatchNorm2d(c),
        #     nn.ReLU(True)
        # )

    def forward(self, x):
        idn = x
        base_c = self.G_channel(x).unsqueeze(2).unsqueeze(3)  # (b, c, 1, 1, k)
        base_h = self.G_high(x.permute(0, 2, 1, 3).contiguous()).unsqueeze(1).unsqueeze(3)    # (b, 1, h, 1, k)
        base_w = self.G_width(x.permute(0, 3, 2, 1).contiguous()).unsqueeze(1).unsqueeze(2)    # (b, 1, 1, w, k)

        attention_map = base_h * base_w * base_c  # (b, c, h, w, k)
        attention_map = torch.sum(attention_map, dim=-1)

        x = attention_map * x
        # x_a = torch.cat((idn, x), dim=1)
        # out = self.conv_out(x_a)
        return x





class TANet(SegBaseModel):

    def __init__(self, n_class, image_size, backbone='resnet34', aux=False, pretrained_base=False, dilated=True, deep_stem=False, **kwargs):
        super(TANet, self).__init__(backbone, pretrained_base=pretrained_base, dilated=dilated, deep_stem=deep_stem, **kwargs)
        self.aux = aux
        self.dilated = dilated
        channels = self.base_channel
        if deep_stem or backbone == 'resnest101':
            conv1_channel = 128
        else:
            conv1_channel = 64

        self.tensor_attention5 = Tensor_Attention(image_size[0]//32, image_size[1]//32, channels[3])
        self.tensor_attention4 = Tensor_Attention(image_size[0] // 16, image_size[1] // 16, channels[2])
        self.tensor_attention3 = Tensor_Attention(image_size[0] // 8, image_size[1] // 8, channels[1])
        self.tensor_attention2 = Tensor_Attention(image_size[0] // 4, image_size[1] // 4, channels[0])

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
        c1, c2, c3, c4, c5 = self.backbone.extract_features(x)

        c2 = self.tensor_attention2(c2)
        c3 = self.tensor_attention3(c3)
        c4 = self.tensor_attention4(c4)
        c5 = self.tensor_attention5(c5)

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























