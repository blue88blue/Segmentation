import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='kaiming'):
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)




class _FCNHead(nn.Module):
    def __init__(self, in_channels, n_class, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        self.n_class = n_class
        # inter_channels = max(in_channels // 4, 16)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, n_class, 1)
        )
    def forward(self, x):
        pred = self.block(x)
        if self.n_class == 1:
            pred = torch.sigmoid(pred)
        return pred

