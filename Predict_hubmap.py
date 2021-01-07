from torch.utils.data import DataLoader
import torch
import os
from PIL import Image
from torch.nn import functional as F
from settings_hubmap import *
import numpy as np
import csv
from tqdm import tqdm
#models
from model.choose_model import seg_model


# #################################### predict settings 预测提交结果 ####################################
pred_data_root = ''
pred_dir = "pred_mask"
map_dir = "pred_map"
submission_file = "submission.csv"
model_dir = "checkpoints/CP_epoch250_84.4923.pth"
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
                    for i in range(image.shape[0]):
                        pred_i = pred[i, ...]
                        # pred_i = F.interpolate(pred_i.unsqueeze(0), size=(int(image_size[i, 0]), int(image_size[i, 1])), mode="bilinear", align_corners=True).squeeze()
                        pred_mask = torch.max(torch.exp(pred_i.cpu()), dim=0)[1]
                        pred_mask = pred_mask.squeeze().numpy()
                        pred_mask = Image.fromarray(np.uint8(pred_mask) * 255).convert("L")
                        pred_mask.save(os.path.join(pred_dir, file_name[i] + ".png"))

                        pred_i = F.softmax(pred_i, dim=0)
                        pred_map = Image.fromarray(np.uint8(pred_i[1, ...].cpu().squeeze().numpy() * 255)).convert("L")
                        pred_map.save(os.path.join(map_dir, file_name[i] + ".png"))
                pbar.update(image.shape[0])

class merge_manger():
    def __init__(self,submissionfile, resized_image_path="/media/sjh/disk1T/dataset/kidney/test_imageresize", information_file="/media/sjh/disk1T/dataset/kidney/HuBMAP-20-dataset_information.csv", crop_size=[256, 256]):
        self.resized_image_path = resized_image_path
        self.crop_size = crop_size
        self.crop_gap = list(np.array(crop_size) * 4 // 5)

        self.test_ids = [
            "26dc41664", "afa5e8098", "b2dc8411c", "b9a3865fc",
            "c68fe75ea", ]
        with open(information_file, "r") as f:
            reader = csv.reader(f)
            self.dataset_inf = list(reader)

        self.submissionfile = submissionfile
        with open(submissionfile, "w") as f:
            w = csv.writer(f)
            w.writerow(["id", "predicted"])


    def get_positions(self, h, w):
        crop_positions = []
        position = [0, self.crop_size[0], 0, self.crop_size[1]]
        while (1):
            if position[1] > h:  # or position[1] > h * 3 / 4:
                position[1] = h
                position[0] = h - self.crop_size[0]
            position[2] = 0
            position[3] = self.crop_size[1]
            while (1):
                if position[3] > w:  # or position[3] > w * 3 / 4:
                    position[3] = w
                    position[2] = w - self.crop_size[1]
                # 记录所有裁剪位置
                crop_positions.append(position.copy())

                if position[3] == w:
                    break
                position[2] += self.crop_gap[1]
                position[3] += self.crop_gap[1]

            if position[1] == h:
                break
            position[0] += self.crop_gap[0]
            position[1] += self.crop_gap[0]

        return crop_positions

    def merge_map(self, map_dir):
        merge_map_dir = map_dir+"_merge_result"
        if not os.path.exists(merge_map_dir):
            os.mkdir(merge_map_dir)

        for image_id in self.test_ids:
            print(image_id)
            image_w, image_h = Image.open(os.path.join(self.resized_image_path, image_id+".png")).size
            score_map = np.zeros((image_h, image_w, 2), dtype= np.int16)
            crop_positions = self.get_positions(image_h, image_w)

            for i, position in enumerate(crop_positions):
                crop_score_map = np.array(Image.open(os.path.join(map_dir, image_id + f"_{i}.png")))
                crop_score_map = np.expand_dims(crop_score_map, axis=2)
                crop_score_map_1 = crop_score_map
                crop_score_map_0 = 255 - crop_score_map
                crop_score_map = np.concatenate((crop_score_map_0, crop_score_map_1), axis=2)
                score_map[position[0]:position[1], position[2]:position[3], :] += crop_score_map
            mask = np.argmax(score_map, axis=2)
            origin_size = self.get_size_from_dataset_inf(image_id)
            origin_size_mask = F.interpolate(torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0), size=origin_size)
            origin_size_mask = origin_size_mask.squeeze().int().numpy()
            enc = self.mask2enc(origin_size_mask)
            with open(self.submissionfile, "a") as f:
                w = csv.writer(f)
                w.writerow([image_id, enc])
            # 保存合并的mask
            mask = np.argmax(score_map, axis=2)
            mask = Image.fromarray(np.uint8(mask)*255).convert("L")
            mask.save(os.path.join(merge_map_dir, image_id+"_mask.png"))

    def mask2enc(self, mask, n=1):
        pixels = mask.T.flatten()
        encs = []
        for i in range(1, n + 1):
            p = (pixels == i).astype(np.int8)
            if p.sum() == 0:
                encs.append(np.nan)
            else:
                p = np.concatenate([[0], p, [0]])
                runs = np.where(p[1:] != p[:-1])[0] + 1
                runs[1::2] -= runs[::2]
                encs.append(' '.join(str(x) for x in runs))
        return encs

    def get_size_from_dataset_inf(self, image_id):
        hw = [0, 0]
        for row in self.dataset_inf:
            if image_id in row[0]:
                hw[0] = int(row[2])
                hw[1] = int(row[1])
                break
        return hw



if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = basic_setting()

    pred_dir = os.path.join(args.dir, pred_dir)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    map_dir = os.path.join(args.dir, map_dir)
    if not os.path.exists(map_dir):
        os.mkdir(map_dir)
    model_dir = os.path.join(args.dir, model_dir)
    submission_file = os.path.join(args.dir, submission_file)

    model = seg_model(args)
    model.to(device)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    print("model loaded!")

    pred(model, device, args)
    manger = merge_manger(submission_file, resized_image_path="/media/sjh/disk1T/dataset/kidney/test_imageresize",
                          information_file="/media/sjh/disk1T/dataset/kidney/HuBMAP-20-dataset_information.csv")
    manger.merge_map(map_dir)

