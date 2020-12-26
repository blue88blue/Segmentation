from torch.utils.data import DataLoader
import torch
import os
from PIL import Image
from torch.nn import functional as F
from settings_REFUGE import *
import numpy as np
import csv
from tqdm import tqdm
#models
from model.choose_model import seg_model


# #################################### predict settings 预测提交结果 ####################################
pred_data_root = '/home/sjh/dataset/PLAM/PALM-Validation400'
val_data_roi_file = '/home/sjh/dataset/REFUGE2_ROI_768/val_ROI.csv'
pred_dir = "segmentation"
# map_dir = "segmentation_map"
model_dir = "checkpoints/fold_1/CP_epoch250_78.4423.pth"
crop_size = (512, 512)
# #################################### predict settings 预测提交结果 ####################################


def pred(model, device, args):
    dataset_pred = predict_Dataset(pred_data_root, args.crop_size)
    dataloader_pred = DataLoader(dataset_pred, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,pin_memory=True, drop_last=False)

    with open(val_data_roi_file, "r") as f:
        reader = csv.reader(f)
        roi_list = list(reader)

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(dataset_pred), desc=f'predict', unit='img') as pbar:
            for batch in dataloader_pred:
                image = batch["image"]
                file_name = batch["file_name"]
                image_size = batch["image_size"]
                image = image.to(device, dtype=torch.float32)

                with torch.no_grad():
                    outputs = model(image)
                    pred = outputs['main_out']

                # 保存预测结果
                for i in range(image.shape[0]):
                    recover_sigle_image_size_and_save(pred[i, :, :, :], image_size[i], file_name[i], roi_list)

                pbar.update(image.shape[0])


# 将预测tensor恢复到原图大小
def recover_sigle_image_size_and_save(image, image_size, file_name, roi_list):
    assert len(image.size()) == 3

    image = F.interpolate(image.unsqueeze(0), size=(image_size[1], image_size[0]), mode='bilinear', align_corners=True).squeeze(0)

    for row in roi_list:
        if file_name in row:
            roi_position = row[1:5]
            roi_pad = row[5:9]
            roi_position = [int(x) for x in roi_position]
            roi_pad = [int(x) for x in roi_pad]
            original_h = int(row[-2])
            original_w = int(row[-1])
            break

    # # 裁掉pad
    if roi_pad != [0, 0, 0, 0]:
        image = image[:, roi_pad[0]:crop_size[0]-roi_pad[1], roi_pad[2]:crop_size[1]-roi_pad[3]]
    # pad到原图大小
    resize_pad = [0, 0, 0, 0]
    resize_pad[0] = roi_position[2]
    resize_pad[1] = original_w - roi_position[3]
    resize_pad[2] = roi_position[0]
    resize_pad[3] = original_h - roi_position[1]
    image = F.pad(image, resize_pad, mode='constant', value=0)

    # 阈值处理
    pred_image = torch.exp(image).max(dim=0)[1]
    pred_image = pred_image.cpu().squeeze().numpy()
    pred_image = np.where(pred_image == 0, 255, pred_image)  # 背景
    pred_image = np.where(pred_image == 1, 128, pred_image)  # disc
    pred_image = np.where(pred_image == 2, 0, pred_image)  # cup

    # 保存
    pred_image = Image.fromarray(np.uint8(pred_image)).convert('RGB')
    pred_image.save(os.path.join(pred_dir, file_name) + ".png")

    # map = F.softmax(image, dim=0)*255
    # map = map.permute(1, 2, 0).cpu().numpy()
    # map[:, :, 0] = np.where(map[:, :, 0] == 85, 255, map[:, :, 0])  # 背景
    # map[:, :, 1] = np.where(map[:, :, 1] == 85, 0, map[:, :, 1])  # disc
    # map[:, :, 2] = np.where(map[:, :, 2] == 85, 0, map[:, :, 2])  # cup
    # map = Image.fromarray(np.uint8(map)).convert('RGB')
    # map.save(os.path.join(map_dir[num_fold], file_name) + ".png")







if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = basic_setting()

    pred_dir = os.path.join(args.dir, pred_dir)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    model_dir = os.path.join(args.dir, model_dir)

    model = seg_model(args)
    model.to(device)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    print("model loaded!")

    pred(model, device, args)