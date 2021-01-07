from torch.utils.data import DataLoader
import torch
import os
from PIL import Image
from torch.nn import functional as F
from settings_PALM import *
import numpy as np
import csv
from tqdm import tqdm
#models
from model.choose_model import seg_model


# #################################### predict settings 预测提交结果 ####################################
pred_data_root = '/home/sjh/dataset/PLAM/PALM-Validation400'
pred_dir = "Atrophy"
model_dir = "checkpoints/fold_2/CP_epoch200_85.9223.pth"
# #################################### predict settings 预测提交结果 ####################################


def pred(model, device, args):
    dataset_pred = predict_Dataset(pred_data_root, args.crop_size)
    dataloader_pred = DataLoader(dataset_pred, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,pin_memory=True, drop_last=False)

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
                    recover_sigle_image_size_and_save(pred[i, :, :, :], args.crop_size, image_size[i], file_name[i])

                pbar.update(image.shape[0])


# 将预测tensor恢复到原图大小
def recover_sigle_image_size_and_save(image, crop_size, image_size, file_name):
    assert len(image.size())==3

    ratio_h = crop_size[0] / float(image_size[0])   # 高度比例
    ratio_w = crop_size[1] / float(image_size[1])   # 宽度比例
    ratio = min(ratio_h, ratio_w)
    h = int(image_size[0] *ratio)
    w = int(image_size[1] *ratio)
    # 裁剪去掉pad
    image_crop = image[:, 0:h, 0:w].unsqueeze(0)
    image = F.interpolate(image_crop, size=(image_size[0], image_size[1]), mode="bilinear", align_corners=True)
    # 阈值处理
    pred_image = torch.exp(image).max(dim=1)[1]
    pred_image = pred_image.cpu().squeeze().numpy()
    pred_image = (1 - np.array(pred_image, dtype=np.int8))*255
    # 保存
    pred_image = Image.fromarray(pred_image).convert('L')
    pred_image.save(os.path.join(pred_dir, file_name) + ".bmp")





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