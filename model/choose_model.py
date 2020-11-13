from model.Unet import Unet
from model.AttUnet import AttUnet
from model.PSPNet import PSPNet
from model.DeepLabV3 import DeepLabV3
from model.DANet import DANet
from model.CPFNet import CPFNet
from model.AGNet.model import AG_Net
from model.my_model.ResUnet import ResUnet
from model.cenet import CE_Net_
from model.my_model.EMANet import EMANet
from model.deeplabv3_plus import DeepLabV3Plus
from model.my_model.EfficientFCN import EfficientFCN
from model.my_model.ccr import EMUPNet
from model.my_model.CaCNet import CaCNet
from model.my_model.Border_ResUnet import Border_ResUnet
from model.my_model.TANet import TANet
from model.my_model.CMSINet import CMSINet

def seg_model(args):
    if args.network == "Unet":
        model = Unet(args.in_channel, args.n_class, channel_reduction=args.Ulikenet_channel_reduction, aux=args.aux)
    elif args.network == "AttUnet":
        model = AttUnet(args.in_channel, args.n_class, channel_reduction=args.Ulikenet_channel_reduction, aux=args.aux)
    elif args.network == "PSPNet":
        model = PSPNet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "DeepLabV3":
        model = DeepLabV3(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "DANet":
        model = DANet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "CPFNet":
        model = CPFNet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "AG_Net":
        model = AG_Net(args.n_class)
    elif args.network == "CENet":
        model = CE_Net_(args.n_class)
    elif args.network == "ResUnet":
        model = ResUnet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "EMANet":  # 增强 PPM + EMAU + sematic flow
        model = EMANet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem, crop_size=args.crop_size)
    elif args.network == "DeepLabV3Plus":
        model = DeepLabV3Plus(args.n_class)
    elif args.network == "EfficientFCN":
        model = EfficientFCN(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=False, deep_stem=args.deep_stem)
    elif args.network == "EMUPNet":
        model = EMUPNet(args.n_class, args.crop_size, args.backbone, pretrained_base=args.pretrained,  deep_stem=args.deep_stem)
    elif args.network == "CaCNet":
        model = CaCNet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "Border_ResUnet":
        model = Border_ResUnet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "TANet":
        model = TANet(args.n_class, args.crop_size, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    elif args.network == "CMSINet":
        model = CMSINet(args.n_class, args.backbone, aux=args.aux, pretrained_base=args.pretrained, dilated=args.dilated, deep_stem=args.deep_stem)
    else:
        NotImplementedError("not implemented {args.network} model")

    return model














