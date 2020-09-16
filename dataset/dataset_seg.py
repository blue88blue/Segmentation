import torch
import os
from torch.utils.data import Dataset
import csv
from .transform import*
import numpy as np


class AI_Dataset(Dataset):
    def __init__(self, data_root, target_root, crop_size, data_mode, k_fold=None, imagefile_csv=None, num_fold=None):
        super().__init__()
        self.crop_size = crop_size  # h, w
        self.data_root = data_root
        self.target_root = target_root
        self.data_mode = data_mode
        # 若不交叉验证，直接读取data_root下文件列表
        if k_fold == None:
            self.image_files = sorted(os.listdir(data_root))
            print(f"{data_mode} dataset: {len(self.image_files)}")
        # 交叉验证：传入包含所有数据集文件名的csv， 根据本次折数num_fold获取文件名列表
        else:
            with open(imagefile_csv, "r") as f:
                reader = csv.reader(f)
                image_files = list(reader)[0]
            image_files = image_files[:10000]###################################
            fold_size = len(image_files) // k_fold  # 等分
            fold = num_fold - 1
            if data_mode == "train":
                self.image_files = image_files[0: fold*fold_size] + image_files[fold*fold_size+fold_size:]
            elif data_mode == "val" or data_mode == "test":
                self.image_files = image_files[fold*fold_size: fold*fold_size+fold_size]
            else:
                raise NotImplementedError
            print(f"{data_mode} dataset fold{num_fold}/{k_fold}: {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = self.image_files[idx]
        file_name = os.path.splitext(file)[0]

        image_path = os.path.join(self.data_root, file)
        label_path = os.path.join(self.target_root, file_name+".png")
        image, label = fetch(image_path, label_path)


        if self.data_mode == "train":  # 数据增强
            image, label = random_transfrom(image, label)

        image, label = convert_to_tensor(image, label)

        # -------标签处理-------
        label = label/100 - 1
        # -------标签处理-------

        if self.data_mode == "train":  # 数据增强
            image, label = random_Top_Bottom_filp(image, label)
            image, label = random_Left_Right_filp(image, label)

        # image, label = resize(self.crop_size, image, label)
        label = label.squeeze(0)

        return{
            "image": image,
            "label": label,
            "file": file}



class AI_PredictDataset(Dataset):
    def __init__(self, data_root, crop_size):
        super(AI_PredictDataset, self).__init__()
        self.data_root = data_root
        self.crop_size = crop_size

        self.files = os.listdir(data_root)
        self.files = sorted(self.files)
        print(f"pred dataset: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name, _ = os.path.splitext(self.files[idx])
        image_path = os.path.join(self.data_root, self.files[idx])

        image, _ = fetch(image_path)
        image, _ = convert_to_tensor(image)
        # image, _ = resize(self.crop_size, image)

        return {
            "image": image,
            "file_name": file_name}


