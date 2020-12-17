from settings_PALM import *
import torch.nn.functional as F
from utils import utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from utils.loss import DiceLoss, OhemCrossEntropy, OhemCrossEntropy_per_image
from tqdm import tqdm
import csv
import random
import numpy as np
import sys
from PIL import Image
import time
import torchsummary
from torchvision.utils import save_image
# models
from model.choose_model import seg_model


def main(args, num_fold=0):
    # 模型选择
    model = seg_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # if args.mode == "train" and num_fold <= 1:
    #     torchsummary.summary(model, (3, args.crop_size[0], args.crop_size[1]))  # #输出网络结构和参数量
    print(f'   [network: {args.network}  device: {device}]')

    if args.mode == "train":
        try:
            train(model, device, args, num_fold=num_fold)
        except KeyboardInterrupt:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir[num_fold], 'INTERRUPTED.pth'))
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    elif args.mode == "test":
        if args.k_fold is not None:
            return test(model, device, args, num_fold=num_fold)
        else:
            test(model, device, args, num_fold=num_fold)
    else:
        raise NotImplementedError





def train(model, device, args, num_fold=0):
    dataset_train = myDataset(args.data_root, args.target_root, args.crop_size,  "train",
                                 k_fold=args.k_fold, imagefile_csv=args.dataset_file_list, num_fold=num_fold)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    num_train_data = len(dataset_train)  # 训练数据大小
    dataset_val = myDataset(args.data_root, args.target_root, args.crop_size, "val",
                               k_fold=args.k_fold, imagefile_csv=args.dataset_file_list, num_fold=num_fold)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
    num_train_val = len(dataset_val)  # 验证数据大小
    ####################
    writer = SummaryWriter(log_dir=args.log_dir[num_fold], comment=f'tb_log')

    if args.optim == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 定义损失函数
    if args.OHEM:
        criterion = OhemCrossEntropy(thres=0.8, min_kept=10000)
    else:
        criterion = nn.CrossEntropyLoss(torch.tensor(args.class_weight, device=device))
    criterion_dice = DiceLoss()

    cp_manager = utils.save_checkpoint_manager(3)
    step = 0
    for epoch in range(args.num_epochs):
        model.train()
        lr = utils.poly_learning_rate(args, opt, epoch)  # 学习率调节

        with tqdm(total=num_train_data, desc=f'[Train] fold[{num_fold}/{args.k_fold}] Epoch[{epoch + 1}/{args.num_epochs} LR{lr:.8f}] ', unit='img') as pbar:
            for batch in dataloader_train:
                step += 1
                # 读取训练数据
                image = batch["image"]
                label = batch["label"]
                assert len(image.size()) == 4
                assert len(label.size()) == 3
                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)

                # 前向传播
                opt.zero_grad()
                outputs = model(image)
                main_out = outputs["main_out"]

                # 计算损失
                diceloss = criterion_dice(main_out, label)
                celoss = criterion(main_out, label)
                totall_loss = celoss + diceloss * args.dice_weight
                if "sim_loss" in outputs.keys():
                    totall_loss += outputs["sim_loss"]*0.2
                if "aux_out" in outputs.keys():  # 计算辅助损失函数
                    aux_losses = 0
                    for aux_p in outputs["aux_out"]:
                        auxloss = (criterion(aux_p, label) + criterion_dice(aux_p, label) * args.dice_weight) * args.aux_weight
                        totall_loss += auxloss
                        aux_losses += auxloss

                if "mu" in outputs.keys():  # EMAU 的基更新
                    with torch.no_grad():
                        mu = outputs["mu"]
                        mu = mu.mean(dim=0, keepdim=True)
                        momentum = 0.9
                        # model.emau.mu *= momentum
                        # model.emau.mu += mu * (1 - momentum)
                        model.effcient_module.em.mu *= momentum
                        model.effcient_module.em.mu += mu * (1 - momentum)
                if "mu1" in outputs.keys():
                    with torch.no_grad():
                        mu1 = outputs['mu1'].mean(dim=0, keepdim=True)
                        model.donv_up1.em.mu = model.donv_up1.em.mu * 0.9 + mu1 * (1 - 0.9)

                        mu2 = outputs['mu2'].mean(dim=0, keepdim=True)
                        model.donv_up2.em.mu = model.donv_up2.em.mu * 0.9 + mu2 * (1 - 0.9)

                        mu3 = outputs['mu3'].mean(dim=0, keepdim=True)
                        model.donv_up3.em.mu = model.donv_up3.em.mu * 0.9 + mu3 * (1 - 0.9)

                        mu4 = outputs['mu4'].mean(dim=0, keepdim=True)
                        model.donv_up4.em.mu = model.donv_up4.em.mu * 0.9 + mu4 * (1 - 0.9)
                totall_loss.backward()
                opt.step()

                if step % 5 == 0:
                    writer.add_scalar("Train/CE_loss", celoss.item(), step)
                    writer.add_scalar("Train/Dice_loss", diceloss.item(), step)
                    if args.aux:
                        writer.add_scalar("Train/aux_losses",aux_losses, step)
                    if "sim_loss" in outputs.keys():
                        writer.add_scalar("Train/sim_loss", outputs["sim_loss"], step)
                    writer.add_scalar("Train/Totall_loss", totall_loss.item(), step)

                pbar.set_postfix(**{'loss': totall_loss.item()})  # 显示loss
                pbar.update(image.size()[0])


        if (epoch+1) % args.val_step == 0:
            # 验证
            mDice, mIoU, mAcc, mSensitivity, mSpecificity = val(model, dataloader_val, num_train_val, device, args)
            writer.add_scalar("Valid/Dice_val", mDice, step)
            writer.add_scalar("Valid/IoU_val", mIoU, step)
            writer.add_scalar("Valid/Acc_val", mAcc, step)
            writer.add_scalar("Valid/Sen_val", mSensitivity, step)
            writer.add_scalar("Valid/Spe_val", mSpecificity, step)
            # 写入csv文件
            val_result = [num_fold, epoch+1, mDice, mIoU, mAcc, mSensitivity, mSpecificity]
            with open(args.val_result_file, "a") as f:
                w = csv.writer(f)
                w.writerow(val_result)
            # 保存模型
            cp_manager.save(model, os.path.join(args.checkpoint_dir[num_fold], f'CP_epoch{epoch + 1}_{float(mDice):.4f}.pth'), float(mDice))
            if (epoch + 1) == (args.num_epochs):
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir[num_fold], f'CP_epoch{epoch + 1}_{float(mDice):.4f}.pth'))



def val(model, dataloader, num_train_val,  device, args):
    all_dice = []
    all_iou = []
    all_acc = []
    all_sen = []
    all_spe = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=num_train_val, desc=f'VAL', unit='img') as pbar:
            for batch in dataloader:
                image = batch["image"]
                label = batch["label"]
                assert len(image.size()) == 4
                assert len(label.size()) == 3
                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)

                outputs = model(image)
                main_out = outputs["main_out"]
                main_out = torch.exp(main_out).max(dim=1)[1]  # 阈值处理

                # 逐图片计算指标
                for b in range(image.size()[0]):
                    hist = utils.fast_hist(label[b,:,:], main_out[b,:,:], args.n_class)
                    dice, iou, acc, Sensitivity, Specificity = utils.cal_scores(hist.cpu().numpy())
                    all_dice.append(list(dice))
                    all_iou.append(list(iou))
                    all_acc.append([acc])
                    all_sen.append(list(Sensitivity))
                    all_spe.append(list(Specificity))
                pbar.update(image.size()[0])
    # 验证集指标
    mDice = np.array(all_dice).mean()
    mIoU = np.array(all_iou).mean()
    mAcc = np.array(all_acc).mean()
    mSensitivity = np.array(all_sen).mean()
    mSpecificity = np.array(all_spe).mean()

    dice = list(np.array(all_dice).mean(axis=0))
    print(f"dice{dice}")
    print(f'\r   [VAL] mDice:{mDice:0.2f}, mIoU:{mIoU:0.2f}, mAcc:{mAcc:0.2f}, mSen:{mSensitivity:0.2f}, mSpec:{mSpecificity:0.2f}')

    return mDice, mIoU, mAcc, mSensitivity, mSpecificity



def test(model, device, args, num_fold=0):
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

    dataset_test = myDataset(args.data_root, args.target_root, args.crop_size, "test",
                                k_fold=args.k_fold, imagefile_csv=args.dataset_file_list, num_fold=num_fold)
    dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    all_dice = []
    all_iou = []
    all_acc = []
    all_sen = []
    all_spe = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(dataset_test), desc=f'TEST fold {num_fold}/{args.k_fold}', unit='img') as pbar:
            for batch in dataloader:
                image = batch["image"]
                label = batch["label"]
                file = batch["file"]
                assert len(image.size()) == 4
                assert len(label.size()) == 3
                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)

                outputs = model(image)
                pred = outputs["main_out"]
                pred = torch.exp(pred).max(dim=1)[1]  # 阈值处理

                for b in range(image.size()[0]):
                    hist = utils.fast_hist(label[b,:,:], pred[b,:,:], args.n_class)
                    dice, iou, acc, Sensitivity, Specificity = utils.cal_scores(hist.cpu().numpy(), smooth=0.01)

                    # 写入每个测试数据的指标
                    test_result = [file[b], dice.mean()]+list(dice)+[iou.mean()]+list(iou)+[acc] + \
                        [Sensitivity.mean()]+list(Sensitivity)+[Specificity.mean()]+list(Specificity)
                    with open(args.test_result_file, "a") as f:
                        w = csv.writer(f)
                        w.writerow(test_result)

                    all_dice.append(list(dice))
                    all_iou.append(list(iou))
                    all_acc.append([acc])
                    all_sen.append(list(Sensitivity))
                    all_spe.append(list(Specificity))
                    if args.plot:
                        file_name, _ = os.path.splitext(file[b])
                        # save_image(pred[b,:,:].cpu().float().unsqueeze(0), os.path.join(args.plot_save_dir, file_name + f"_pred_{dice.mean():.2f}.png"), normalize=True)
                        # save_image(image[b,:,:].cpu(), os.path.join(args.plot_save_dir, file[b]))
                        # save_image(label[b,:,:].cpu().float().unsqueeze(0), os.path.join(args.plot_save_dir, file_name + f"_label.png"), normalize=True)
                        if "A4" in outputs.keys():
                            for i in range(0, 25, 5):
                                proj_map = F.interpolate(outputs["A1"][b, ...].unsqueeze(0), size=image.size()[-2:],
                                                      mode="bilinear", align_corners=True).squeeze(0)
                                save_image(proj_map[i, :, :], os.path.join(args.plot_save_dir, file_name + f"_A1_{i}.png"),
                                           normalize=True)
                            for i in range(0, 25, 5):
                                proj_map = F.interpolate(outputs["A2"][b, ...].unsqueeze(0), size=image.size()[-2:],
                                                      mode="bilinear", align_corners=True).squeeze(0)
                                save_image(proj_map[i, :, :], os.path.join(args.plot_save_dir, file_name + f"_A2_{i}.png"),
                                           normalize=True)
                            for i in range(0, 25, 5):
                                proj_map = F.interpolate(outputs["A3"][b, ...].unsqueeze(0), size=image.size()[-2:],
                                                      mode="bilinear", align_corners=True).squeeze(0)
                                save_image(proj_map[i, :, :], os.path.join(args.plot_save_dir, file_name + f"_A3_{i}.png"),
                                           normalize=True)
                            for i in range(0, 25, 5):
                                proj_map = F.interpolate(outputs["A4"][b, ...].unsqueeze(0), size=image.size()[-2:],
                                                      mode="bilinear", align_corners=True).squeeze(0)
                                save_image(proj_map[i, :, :], os.path.join(args.plot_save_dir, file_name + f"_A4_{i}.png"),
                                           normalize=True)
                        # if "x_proj_1" in outputs.keys():
                        #     for i in range(7):
                        #         proj_map = F.interpolate(outputs["x_proj_1"][b, ...].unsqueeze(0), size=image.size()[-2:],
                        #                               mode="bilinear", align_corners=True).squeeze(0)
                        #         save_image(proj_map[i, :, :], os.path.join(args.plot_save_dir, file_name + f"_A1_{i}.png"),
                        #                    normalize=True)
                        #     for i in range(7):
                        #         proj_map = F.interpolate(outputs["x_proj_2"][b, ...].unsqueeze(0), size=image.size()[-2:],
                        #                               mode="bilinear", align_corners=True).squeeze(0)
                        #         save_image(proj_map[i, :, :], os.path.join(args.plot_save_dir, file_name + f"_A2_{i}.png"),
                        #                    normalize=True)
                        save_image(image[b, :, :].cpu(), os.path.join(args.plot_save_dir, file[b]))
                        pred_image = pred[b, :, :].unsqueeze(0)
                        true_mask = label[b, :, :].unsqueeze(0)
                        result_image = torch.cat((pred_image, true_mask, torch.zeros_like(pred_image)), dim=0).permute(1, 2, 0).cpu().numpy()
                        result_image = Image.fromarray(np.uint8(result_image) * 255)
                        result_image.save(args.plot_save_dir + f"/{file_name}({dice.mean():.2f}).png")
                pbar.update(image.size()[0])

    print(f"\r---------Fold {num_fold} Test Result---------")
    print(f'mDice: {np.array(all_dice).mean()}')
    print(f'mIoU:  {np.array(all_iou).mean()}')
    print(f'mAcc:  {np.array(all_acc).mean()}')
    print(f'mSens: {np.array(all_sen).mean()}')
    print(f'mSpec: {np.array(all_spe).mean()}')

    if num_fold == 0:
        utils.save_print_score(all_dice, all_iou, all_acc, all_sen, all_spe, args.test_result_file, args.label_names)
        return

    return all_dice, all_iou, all_acc, all_sen, all_spe



if __name__ == "__main__":

    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    # cudnn.benchmark = True

    args = basic_setting()
    assert args.k_fold != 1
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id

    # 交叉验证所需， 文件名列表
    if (not os.path.exists(args.dataset_file_list)) and (args.k_fold is not None):
        utils.get_dataset_filelist(args.data_root, args.dataset_file_list)

    mode = args.mode
    if args.k_fold is None:
        print("k_fold is None")
        if mode == "train_test":
            args.mode = "train"
            print("###################### Train Start ######################")
            main(args)
            args.mode = "test"
            print("###################### Test Start ######################")
            main(args)
        else:
            main(args)
    else:
        if mode == "train_test":
            print("###################### Train & Test Start ######################")

        if mode == "train" or mode == "train_test":
            args.mode = "train"
            print("###################### Train Start ######################")
            for i in range(args.start_fold, args.end_fold):
                torch.cuda.empty_cache()
                main(args, num_fold=i + 1)

        if mode == "test" or mode == "train_test":
            args.mode = "test"
            print("###################### Test Start ######################")
            all_dice = []
            all_iou = []
            all_acc = []
            all_sen = []
            all_spe = []
            for i in range(args.start_fold, args.end_fold):
                Dice, IoU, Acc, Sensitivity, Specificity = main(args, num_fold=i + 1)
                all_dice += Dice
                all_iou += IoU
                all_acc += Acc
                all_sen += Sensitivity
                all_spe += Specificity
            utils.save_print_score(all_dice, all_iou, all_acc, all_sen, all_spe, args.test_result_file, args.label_names)







