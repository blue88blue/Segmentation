from model.segbase import SegBaseModel
from model.model_utils import init_weights, _FCNHead
from .blocks import *
from torch.utils.data import WeightedRandomSampler
from utils.utils import make_one_hot


class hard_contrast_loss(nn.Module):

    def __init__(self, n_class, k=500):
        super(hard_contrast_loss, self).__init__()
        self.k = k
        self.n_class = n_class

    def forward(self, x, logit, label):
        b, c, h, w = x.size()
        x = F.interpolate(x, size=logit.size()[-2:], mode="bilinear", align_corners=True)
        if self.k > h*w:
            self.k = h*w

        pred = F.softmax(logit, dim=1)
        pred_mask = torch.exp(pred).max(dim=1)[1]  # 阈值处理
        onehot_pred = make_one_hot(pred_mask, self.n_class).bool()
        onehot_label = make_one_hot(label, self.n_class).bool()
        TP = (onehot_pred & onehot_label)
        TN = (~onehot_pred & ~onehot_label)
        FP = (onehot_pred & ~onehot_label)
        FN = (~onehot_pred & onehot_label)
        f = self._l2norm(x.reshape(b, c, -1), dim=1)
        loss = []
        for c in range(self.n_class):
            TP_base = []
            TN_base = []
            FP_base = []
            FN_base = []
            for batch in range(b):
                TP_samples = list(WeightedRandomSampler(list(TP[batch, c, :, :].reshape(-1)), int(self.k*0.35), replacement=True))
                TN_samples = list(WeightedRandomSampler(list(TN[batch, c, :, :].reshape(-1)), int(self.k*0.35), replacement=True))
                FP_samples = list(WeightedRandomSampler(list(FP[batch, c, :, :].reshape(-1)), int(self.k*0.15), replacement=True))
                FN_samples = list(WeightedRandomSampler(list(FN[batch, c, :, :].reshape(-1)), int(self.k*0.15), replacement=True))
                TP_base.append(f[batch, :, TP_samples])
                TN_base.append(f[batch, :, TN_samples])
                FP_base.append(f[batch, :, FP_samples])
                FN_base.append(f[batch, :, FN_samples])
            print(torch.cat(TP_base, dim=1).size())
            TP_base = torch.cat(TP_base, dim=1)  # (c, N)
            TN_base = torch.cat(TN_base, dim=1)  # (c, N)
            FP_base = torch.cat(FP_base, dim=1).permute(1, 0)  # (N, c)
            FN_base = torch.cat(FN_base, dim=1).permute(1, 0)  # (N, c)

            FP_TP = -1*torch.mm(FP_base, TP_base)  # -> 1
            FP_TN = torch.mm(FP_base, TN_base)  # -> -1
            FN_TP = torch.mm(FN_base, TP_base)  # -> -1
            FN_TN = -1*torch.mm(FN_base, TN_base)  # -> 1

            loss.append(torch.cat((FP_TP, FP_TN, FN_TP, FN_TN), dim=0).mean())
        loss = torch.cat(loss).mean()
        return loss

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


