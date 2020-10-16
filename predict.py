from dataset.dataset_AI import predict_Dataset
from dataset.transform import scale_adaptive
from torch.utils.data import DataLoader
import torch
import os
from PIL import Image
from torch.nn import functional as F
from settings_PALM import basic_setting
import numpy as np
import csv
from tqdm import tqdm
#models
from model.choose_model import seg_model

# #################################### predict settings 预测提交结果 ####################################
pred_data_root = "/home/sjh/dataset/AI+/image_A"  # 预测图片路径
pred_dir = "segmentation"     # mask
map_dir = "segmentation_map"  # 概率图
model_CPepoch = 30
# #################################### predict settings 预测提交结果 ####################################


def pred(model, device, args, num_fold=0):
    dataset_pred = predict_Dataset(pred_data_root, args.crop_size)
    dataloader_pred = DataLoader(dataset_pred, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,pin_memory=True, drop_last=True)
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(dataset_pred), desc=f'predict', unit='img') as pbar:
            for batch in dataloader_pred:
                image = batch["image"]
                file_name = batch["file_name"]

                image = image.to(device, dtype=torch.float32)
                outputs = model(image)
                pred = outputs['main_out']
                # pred = F.interpolate(pred, size=(256, 256), mode='bilinear', align_corners=True)

                for i in range(image.shape[0]):
                    pred_i = pred[i, ...]
                    pred_mask = torch.max(torch.exp(pred_i.squeeze().cpu()), dim=0)[1]
                    pred_mask = pred_mask.squeeze().numpy()
                    pred_mask = Image.fromarray(np.uint16(pred_mask+1)*100)
                    pred_mask.save(os.path.join(pred_dir[num_fold], file_name[i] + ".png"))


                pbar.update(image.shape[0])



if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = basic_setting()

    # 模型选择
    model = seg_model(args)

    pred_dir = [os.path.join(args.dir, pred_dir)]
    map_dir = [os.path.join(args.dir, map_dir)]
    if not os.path.exists(pred_dir[0]):
        os.mkdir(pred_dir[0])
    if not os.path.exists(map_dir[0]):
        os.mkdir(map_dir[0])

    if args.k_fold == None:
        model.to(device)
        model_dir = os.path.join(args.checkpoint_dir[0], f'CP_epoch{model_CPepoch}.pth')  # 最后一个epoch模型
        model.load_state_dict(torch.load(model_dir, map_location=device))
        print("model loaded!")

        pred(model, device, args)

    else:
        for i in range(args.start_fold, args.end_fold):
            pred_dir.append(os.path.join(args.dir, f'segmentation_{i+1}'))
            if not os.path.exists(pred_dir[-1]):
                os.mkdir(pred_dir[-1])
            map_dir.append(os.path.join(args.dir, f'segmentation_map{i + 1}'))
            if not os.path.exists(map_dir[-1]):
                os.mkdir(map_dir[-1])
            model.to(device)
            model_dir = os.path.join(args.checkpoint_dir[i+1], f'CP_epoch{model_CPepoch}.pth')  # 最后一个epoch模型
            model.load_state_dict(torch.load(model_dir, map_location=device))
            print("model loaded!")

            pred(model, device, args, i+1)
