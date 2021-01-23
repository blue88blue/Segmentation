import os
import torch
import csv
# from medpy import metric
from tqdm import tqdm
from utils import utils
import SimpleITK as sitk
from torch.utils.data import DataLoader
from dataset.dataset_RETOUCH import volume_Dataset
from dataset.transform import *

def val_3D(model, device, args, num_fold):
    data_root = args.val_data_root if hasattr(args, "val_data_root") else args.data_root
    target_root = args.val_target_root if hasattr(args, "val_target_root") else args.data_root
    val_image_root = os.path.join(data_root, f"f{num_fold}")
    val_mask_root = os.path.join(target_root, f"f{num_fold}")
    volumes = os.listdir(val_image_root)
    all_dice = []
    all_iou = []
    all_acc = []
    all_sen = []
    all_spe = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(volumes), desc=f'VAL', unit='volume') as pbar:
            for volume in volumes:
                volume_img_path = os.path.join(val_image_root, volume)
                volume_mask_path = os.path.join(val_mask_root, volume)
                # print(volume_img_path, volume_mask_path)
                v_dataset = volume_Dataset(volume_img_path, volume_mask_path, args.crop_size)
                v_dataloader = DataLoader(v_dataset, batch_size=1, shuffle=False,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=False)
                volume_pred = []
                volume_label = []
                for batch in v_dataloader:
                    image = batch["image"]
                    label = batch["label"]
                    image = image.to(device, dtype=torch.float32)
                    label = label.to(device, dtype=torch.long)

                    outputs = model(image)
                    main_out = outputs["main_out"]
                    main_out = torch.exp(main_out).max(dim=1)[1]  # 阈值处理
                    volume_pred.append(main_out)
                    volume_label.append(label)
                volume_pred = torch.cat(volume_pred, dim=0)
                volume_label = torch.cat(volume_label, dim=0)

                hist = utils.fast_hist(volume_label, volume_pred, args.n_class)
                dice, iou, acc, Sensitivity, Specificity = utils.cal_scores(hist.cpu().numpy(), drop_non=args.drop_non)
                all_dice.append(list(dice))
                all_iou.append(list(iou))
                all_acc.append([acc])
                all_sen.append(list(Sensitivity))
                all_spe.append(list(Specificity))

                pbar.update(1)

    volume_pred = sitk.GetImageFromArray(np.array(volume_pred.cpu().numpy(), np.int8))
    sitk.WriteImage(volume_pred, os.path.join(args.plot_save_dir, volume + '.nii.gz'))
    # 验证集指标
    if args.drop_non:
        mDice = np.mean(np.array(all_dice).sum(axis=0)/np.sum(np.array(all_dice) > 0, axis=0))
        mIoU =np.mean(np.array(all_iou).sum(axis=0)/np.sum(np.array(all_iou) > 0, axis=0))
        mAcc = np.array(all_acc).mean()
        mSensitivity = np.mean(np.array(all_sen).sum(axis=0)/np.sum(np.array(all_sen) > 0, axis=0))
        mSpecificity = np.mean(np.array(all_spe).sum(axis=0)/np.sum(np.array(all_spe) > 0, axis=0))
    else:
        mDice = np.array(all_dice).mean()
        mIoU = np.array(all_iou).mean()
        mAcc = np.array(all_acc).mean()
        mSensitivity = np.array(all_sen).mean()
        mSpecificity = np.array(all_spe).mean()

    dice = list(np.array(all_dice).sum(axis=0)/np.sum(np.array(all_dice) > 0, axis=0))
    print(f"dice{dice}")
    print(f'\r   [VAL] mDice:{mDice:0.2f}, mIoU:{mIoU:0.2f}, mAcc:{mAcc:0.2f}, mSen:{mSensitivity:0.2f}, mSpec:{mSpecificity:0.2f}')

    return mDice, mIoU, mAcc, mSensitivity, mSpecificity







def test_3D(model, device, args, num_fold=0):
    # 导入模型, 选取每一折的最优模型
    if os.path.exists(args.val_result_file):
        with open(args.val_result_file, "r") as f:
            reader = csv.reader(f)
            val_result = list(reader)
        best_epoch = utils.best_model_in_fold(val_result, num_fold)
    else:
        best_epoch = args.num_epochs

    # 导入模型
    model_list = os.listdir(args.checkpoint_dir[num_fold])
    model_dir = [x for x in model_list if str(best_epoch) in x][0]
    model_dir = os.path.join(args.checkpoint_dir[num_fold], model_dir)
    if not os.path.exists(model_dir):
        model_dir = os.path.join(args.checkpoint_dir[num_fold], f'CP_epoch{best_epoch}.pth')
    model.load_state_dict(torch.load(model_dir, map_location=device))
    print(f'\rtest model loaded: [fold:{num_fold}] [best_epoch:{best_epoch}]')


    data_root = args.val_data_root if hasattr(args, "val_data_root") else args.data_root
    target_root = args.val_target_root if hasattr(args, "val_target_root") else args.data_root
    val_image_root = os.path.join(data_root, f"f{num_fold}")
    val_mask_root = os.path.join(target_root, f"f{num_fold}")
    volumes = sorted(os.listdir(val_image_root))
    all_dice = []
    all_iou = []
    all_acc = []
    all_sen = []
    all_spe = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(volumes), desc=f'VAL', unit='img') as pbar:
            for volume in volumes:
                volume_img_path = os.path.join(val_image_root, volume)
                volume_mask_path = os.path.join(val_mask_root, volume)
                v_dataset = volume_Dataset(volume_img_path, volume_mask_path, args.crop_size)
                v_dataloader = DataLoader(v_dataset, batch_size=1, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=True, drop_last=False)
                volume_pred = []
                volume_label = []
                for batch in v_dataloader:
                    image = batch["image"]
                    label = batch["label"]
                    image = image.to(device, dtype=torch.float32)
                    label = label.to(device, dtype=torch.long)

                    outputs = model(image)
                    main_out = outputs["main_out"]
                    main_out = torch.exp(main_out).max(dim=1)[1]  # 阈值处理
                    volume_pred.append(main_out)
                    volume_label.append(label)
                volume_pred = torch.cat(volume_pred, dim=0)
                volume_label = torch.cat(volume_label, dim=0)

                hist = utils.fast_hist(volume_label, volume_pred, args.n_class)
                dice, iou, acc, Sensitivity, Specificity = utils.cal_scores(hist.cpu().numpy(), drop_non=args.drop_non)
                all_dice.append(list(dice))
                all_iou.append(list(iou))
                all_acc.append([acc])
                all_sen.append(list(Sensitivity))
                all_spe.append(list(Specificity))

                # 写入每个测试数据的指标
                test_result = [volume, dice.mean()]+list(dice)+[iou.mean()]+list(iou)+[acc] + \
                    [Sensitivity.mean()]+list(Sensitivity)+[Specificity.mean()]+list(Specificity)
                with open(args.test_result_file, "a") as f:
                    w = csv.writer(f)
                    w.writerow(test_result)

                if args.plot:
                    volume_pred = sitk.GetImageFromArray(np.array(volume_pred.cpu().numpy(), np.int8))
                    sitk.WriteImage(volume_pred, os.path.join(args.plot_save_dir, volume + f'_{dice.mean():.2f}.nii.gz'))

                pbar.update(image.size()[0])

    print(f"\r---------Fold {num_fold} Test Result---------")
    if args.drop_non:
        print(f'mDice: {np.mean(np.array(all_dice).sum(axis=0)/np.sum(np.array(all_dice) > 0, axis=0))}')
        print(f'mIoU:  {np.mean(np.array(all_iou).sum(axis=0)/np.sum(np.array(all_iou) > 0, axis=0))}')
        print(f'mAcc:  {np.array(all_acc).mean()}')
        print(f'mSens: {np.mean(np.array(all_sen).sum(axis=0)/np.sum(np.array(all_sen) > 0, axis=0))}')
        print(f'mSpec: {np.mean(np.array(all_spe).sum(axis=0)/np.sum(np.array(all_spe) > 0, axis=0))}')
    else:
        print(f'mDice: {np.array(all_dice).mean()}')
        print(f'mIoU:  {np.array(all_iou).mean()}')
        print(f'mAcc:  {np.array(all_acc).mean()}')
        print(f'mSens: {np.array(all_sen).mean()}')
        print(f'mSpec: {np.array(all_spe).mean()}')

    if num_fold == 0:
        utils.save_print_score(all_dice, all_iou, all_acc, all_sen, all_spe, args.test_result_file, args.label_names, drop_non=args.drop_non)
        return

    return all_dice, all_iou, all_acc, all_sen, all_spe





def slice2volume(data_root, target_root, dir, num_fold):
    val_image_root = os.path.join(data_root, f"f{num_fold}")
    val_mask_root = os.path.join(target_root, f"f{num_fold}")
    volumes = sorted(os.listdir(val_image_root))
    for volume in volumes:
        volume_img_path = os.path.join(val_image_root, volume)
        volume_mask_path = os.path.join(val_mask_root, volume)
        slices = sorted(os.listdir(volume_img_path))
        volume_image = []
        volume_label = []
        for file in slices:
            image_file = os.path.join(volume_img_path, file)
            label_file = os.path.join(volume_mask_path, file)
            image = np.array(Image.open(image_file))
            label = np.array(Image.open(label_file))

            volume_image.append(np.expand_dims(image, 0))
            volume_label.append(np.expand_dims(label, 0))
        volume_image = np.concatenate(volume_image, axis=0)
        volume_label = np.concatenate(volume_label, axis=0)

        volume_image = sitk.GetImageFromArray(np.array(volume_image))
        sitk.WriteImage(volume_image, os.path.join(dir, volume + '.nii.gz'))
        volume_label = sitk.GetImageFromArray(np.array(volume_label))
        sitk.WriteImage(volume_label, os.path.join(dir, volume + '_GT.nii.gz'))

# slice2volume('/media/sjh/disk1T/dataset/RETOUCH_crop/train_all/img', '/media/sjh/disk1T/dataset/RETOUCH_crop/train_all/mask', '/media/sjh/disk1T/dataset/RETOUCH_crop/volume', 1)
# slice2volume('/media/sjh/disk1T/dataset/RETOUCH_crop/train_all/img', '/media/sjh/disk1T/dataset/RETOUCH_crop/train_all/mask', '/media/sjh/disk1T/dataset/RETOUCH_crop/volume', 2)
# slice2volume('/media/sjh/disk1T/dataset/RETOUCH_crop/train_all/img', '/media/sjh/disk1T/dataset/RETOUCH_crop/train_all/mask', '/media/sjh/disk1T/dataset/RETOUCH_crop/volume', 3)