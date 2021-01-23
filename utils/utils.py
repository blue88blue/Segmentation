
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import csv
import os
import random
import h5py
from PIL import Image
import SimpleITK as sitk

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
def cal_scores(hist, smooth=1, drop_non=False):
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

    # 若不存在某一类， 则该类的指标=0
    if drop_non:
        positive = TP + FN
        mask = positive > 0
        dice *= mask
        iou *= mask
        Sensitivity *= mask
        Specificity *= mask

    return dice[1:]*100, iou[1:]*100, Precision*100, Sensitivity[1:]*100, Specificity[1:]*100



# 保存打印指标
def save_print_score(all_dice, all_iou, all_acc, all_sen, all_spe, file, label_names, drop_non=False):
    all_dice = np.array(all_dice)
    all_iou = np.array(all_iou)
    all_acc = np.array(all_acc)
    all_sen = np.array(all_sen)
    all_spe = np.array(all_spe)
    if drop_non:
        dice_class = all_dice.sum(axis=0)/np.sum(all_dice > 0, axis=0)
        iou_class = all_iou.sum(axis=0) / np.sum(all_iou > 0, axis=0)
        sen_class = all_sen.sum(axis=0) / np.sum(all_sen > 0, axis=0)
        spe_class = all_spe.sum(axis=0) / np.sum(all_spe > 0, axis=0)
        test_mean = ["mean"] + [dice_class.mean()] + list(dice_class) + \
                    [iou_class.mean()] + list(iou_class) + \
                    [all_acc.mean()] + \
                    [sen_class.mean()] + list(sen_class) + \
                    [spe_class.mean()] + list(spe_class)
        test_std = [None]
        print("\n##############Test Result##############")
        print(f'mDice: {dice_class.mean()}')
        print(f'mIoU:  {iou_class.mean()}')
        print(f'mAcc:  {all_acc.mean()}')
        print(f'mSens: {sen_class.mean()}')
        print(f'mSpec: {spe_class.mean()}')
    else:
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
        print("\n##############Test Result##############")
        print(f'mDice: {all_dice.mean()}')
        print(f'mIoU:  {all_iou.mean()}')
        print(f'mAcc:  {all_acc.mean()}')
        print(f'mSens: {all_sen.mean()}')
        print(f'mSpec: {all_spe.mean()}')
    label_names = label_names[1:]
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




# 从验证指标中选择最优的epoch
def best_model_in_fold(val_result, num_fold):
    best_epoch = 0
    best_dice = 0
    for row in val_result:
        if str(num_fold) == row[0]:
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



# 保存训练过程中最大的checkpoint
class save_checkpoint_manager:
    def __init__(self, max_save=5):
        self.checkpoints = {}
        self.max_save = max_save

    def save(self, model, path, score):
        if len(self.checkpoints) < self.max_save:
            self.checkpoints[path] = score
            torch.save(model.state_dict(), path)
        else:
            min_value = min(self.checkpoints.values())
            if score > min_value:
                for i in self.checkpoints.keys():
                    if self.checkpoints[i] == min_value:
                        min_key = i
                        break
                os.remove(min_key)
                self.checkpoints.pop(min_key)
                self.checkpoints[path] = score
                torch.save(model.state_dict(), path)



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




def slices2volume_mask(original_volume_dir, pred_mask_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    volume_filenames = sorted(os.path.splitext(file)[0] for file in  os.listdir(original_volume_dir))
    mask_files = sorted(os.listdir(pred_mask_dir))

    for vfile in volume_filenames:
        volume_mask = []
        slices_files = [mfile for mfile in mask_files if vfile in mfile]
        num = len(slices_files)
        for i in range(num):
            file = vfile+f"_{i}.png"
            image = np.array(np.array(Image.open(os.path.join(pred_mask_dir, file))) == 255, np.uint8)
            image = np.expand_dims(image, axis=0)
            volume_mask.append(image)
        # 保存volume mask
        volume_mask = np.concatenate(volume_mask, axis=0)
        volume_mask = sitk.GetImageFromArray(volume_mask)
        # 设置数据信息
        original_volume = sitk.ReadImage(os.path.join(original_volume_dir, vfile+".nii"))
        volume_mask.SetSpacing(original_volume.GetSpacing())
        volume_mask.SetOrigin(original_volume.GetOrigin())
        volume_mask.SetDirection(original_volume.GetDirection())
        if "ct" in vfile:
            vfile = vfile[:-3]
        vfile = vfile[18:]
        sitk.WriteImage(volume_mask, os.path.join(out_dir, vfile+'.nii.gz'))
        print(vfile)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
