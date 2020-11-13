from torch.utils.data import DataLoader
import torch
import os
from PIL import Image
from torch.nn import functional as F
from settings_COVID19 import *
import numpy as np
import csv
from tqdm import tqdm
#models
from model.choose_model import seg_model

# #################################### predict settings 预测提交结果 ####################################
pred_data_root = "/media/sjh/disk1T/dataset/COVID-19-20_v2/validation_slices/image"  # 预测图片路径
pred_dir = "segmentation"     # mask
model_dir = "/media/sjh/disk1T/COVID19/2020-1107-1429_45_Unet__fold_6/checkpoints/fold_1/CP_epoch4.pth"
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
                outputs = model(image)
                pred = outputs['main_out']

                for i in range(image.shape[0]):
                    pred_i = pred[i, ...]
                    pred_i = F.interpolate(pred_i.unsqueeze(0), size=(int(image_size[i, 0]), int(image_size[i, 1])), mode="bilinear", align_corners=True).squeeze()
                    pred_mask = torch.max(torch.exp(pred_i.cpu()), dim=0)[1]
                    pred_mask = pred_mask.squeeze().numpy()
                    pred_mask = Image.fromarray(np.uint8(pred_mask)*255).convert("L")
                    pred_mask.save(os.path.join(pred_dir, file_name[i] + ".png"))

                pbar.update(image.shape[0])



if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = basic_setting()

    # 模型选择
    model = seg_model(args)

    pred_dir = os.path.join(args.dir, pred_dir)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    model.to(device)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    print("model loaded!")

    pred(model, device, args)
