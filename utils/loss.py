import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class BinaryDiceLoss(nn.Module):

    def __init__(self, smooth=1, p=1.0, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        # print(target.size(), predict.size())
        target = target.contiguous().view(target.shape[0], -1)
        predict = predict.contiguous().view(predict.shape[0], -1)

        intersect = 2.0 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        union = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        dice_loss = torch.sub(1.0, intersect/union)  # (batch_size)

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        elif self.reduction == 'none':
            return dice_loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))




class DiceLoss(nn.Module):
    def __init__(self, ignore_index=0, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        n_class = predict.size()[1]
        predict = F.softmax(predict, dim=1)

        dice = BinaryDiceLoss(**self.kwargs)
        dice_loss = 0
        for i in range(n_class):
            if i == self.ignore_index:
                continue
            predict_i = predict[:, i, :, :]
            target_i = (target == i).float()
            dice_loss += dice(predict_i, target_i)

        if self.ignore_index == None:
            dice_loss /= n_class
        else:
            dice_loss /= n_class-1

        return dice_loss


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.8,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0  # 将忽略的标签数字置零
        pred = pred.gather(1, tmp_target.unsqueeze(1))  # 取出对应标签的交叉熵
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()  # 将每个像素对应标签的置信度排序，从小到大
        min_value = pred[min(self.min_kept, pred.numel() - 1)]  # 最小的预测值
        threshold = max(min_value, self.thresh)  # 阈值

        pixel_losses = pixel_losses[mask][ind]  # 经过排序的像素损失
        pixel_losses = pixel_losses[pred < threshold]  # 小于阈值的像素的损失
        return pixel_losses.mean()


