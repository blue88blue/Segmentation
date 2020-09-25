
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import csv
import os
import random
import h5py


def fast_hist(label_true, label_pred, n_class):
    '''
    :param label_true: 0 ~ n_class (batch, h, w)
    :param label_pred: 0 ~ n_class (batch, h, w)
    :param n_class: 类别数
    :return: 对角线上是每一类分类正确的个数，其他都是分错的个数
    '''

    assert n_class > 1

    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(
        n_class * label_true[mask].int() + label_pred[mask].int(),
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)

    return hist

# 计算指标
def cal_scores(hist, smooth=1):
    TP = np.diag(hist)
    FP = hist.sum(axis=0) - TP
    FN = hist.sum(axis=1) - TP
    TN = hist.sum() - TP - FP - FN
    union = TP + FP + FN

    dice = (2*TP+smooth) / (union+TP+smooth)

    iou = (TP+smooth) / (union+smooth)

    Precision = np.diag(hist).sum() / hist.sum()   # 分类正确的准确率  acc

    Sensitivity = (TP+smooth) / (TP+FN+smooth)  # recall

    Specificity = (TN+smooth) / (FP+TN+smooth)

    return dice[1:]*100, iou[1:]*100, Precision*100, Sensitivity[1:]*100, Specificity[1:]*100



# 保存打印指标
def save_print_score(all_dice, all_iou, all_acc, all_sen, all_spe, file, label_names):
    all_dice = np.array(all_dice)
    all_iou = np.array(all_iou)
    all_acc = np.array(all_acc)
    all_sen = np.array(all_sen)
    all_spe = np.array(all_spe)
    test_mean = ["mean"]+[all_dice.mean()] + list(all_dice.mean(axis=0)) + \
                [all_iou.mean()] + list(all_iou.mean(axis=0)) + \
                [all_acc.mean()] + \
                [all_sen.mean()] + list(all_sen.mean(axis=0)) + \
                [all_spe.mean()] + list(all_spe.mean(axis=0))
    test_std = ["std"]+[all_dice.std()] + list(all_dice.std(axis=0)) + \
               [all_iou.std()] + list(all_iou.std(axis=0)) + \
               [all_acc.std()] + \
               [all_sen.std()] + list(all_sen.std(axis=0)) + \
               [all_spe.std()] + list(all_spe.std(axis=0))
    title = [' ', 'mDice'] + [name + "_dice" for name in label_names] + \
            ['mIoU'] + [name + "_iou" for name in label_names] + \
            ['mAcc'] + \
            ['mSens'] + [name + "_sen" for name in label_names] + \
            ['mSpec'] + [name + "_spe" for name in label_names]
    with open(file, "a") as f:
        w = csv.writer(f)
        w.writerow(["Test Result"])
        w.writerow(title)
        w.writerow(test_mean)
        w.writerow(test_std)

    print("\n##############Test Result##############")
    print(f'mDice: {all_dice.mean()}')
    print(f'mIoU:  {all_iou.mean()}')
    print(f'mAcc:  {all_acc.mean()}')
    print(f'mSens: {all_sen.mean()}')
    print(f'mSpec: {all_spe.mean()}')



# 从验证指标中选择最优的epoch
def best_model_in_fold(val_result, num_fold):
    best_epoch = 0
    best_dice = 0
    for row in val_result:
        if str(num_fold) in row:
            if best_dice < float(row[2]):
                best_dice = float(row[2])
                best_epoch = int(row[1])
    return best_epoch


# 读取数据集目录内文件名，保存至csv文件
def get_dataset_filelist(data_root, save_file):
    file_list = os.listdir(data_root)
    random.shuffle(file_list)
    with open(save_file, 'w') as f:
        w = csv.writer(f)
        w.writerow(file_list)


def poly_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    lr = args.lr * (1 - epoch / args.num_epochs) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr




# one hot转成0,1,2,..这样的标签
def make_class_label(mask):
    b = mask.size()[0]
    mask = mask.view(b, -1)
    class_label = torch.max(mask, dim=-1)[0]
    return class_label


# 把0,1,2...这样的类别标签转化为one_hot
def make_one_hot(targets, num_classes):
    targets = targets.unsqueeze(1)
    label = []
    for i in range(num_classes):
        label.append((targets == i).float())
    label = torch.cat(label, dim=1)
    return label




def save_h5(train_data, train_label, val_data, filename):
    file = h5py.File(filename, 'w')
    # 写入
    file.create_dataset('train_data', data=train_data)
    file.create_dataset('train_label', data=train_label)
    file.create_dataset('val_data', data=val_data)
    file.close()


def load_h5(path):
    file = h5py.File(path, 'r')
    train_data = torch.tensor(np.array(file['train_data'][:]))
    train_label = torch.tensor(np.array(file['train_label'][:]))
    val_data = torch.tensor(np.array(file['val_data'][:]))
    file.close()
    return train_data, train_label, val_data




