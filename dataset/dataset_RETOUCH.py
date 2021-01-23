import torch
import os
from torch.utils.data import Dataset
import csv
from .transform import*
import numpy as np
import glob

mean = torch.Tensor(np.array([0.2294587]))
std = torch.Tensor(np.array([0.10523194]))



class myDataset(Dataset):
    def __init__(self, data_root, target_root, crop_size, data_mode, k_fold=3, imagefile_csv=None, num_fold=None):
        super().__init__()
        self.crop_size = crop_size  # h, w
        self.data_root = data_root
        self.target_root = target_root
        self.data_mode = data_mode
        a = len(data_root)+1

        if data_mode == "train":
            image_files = []
            for i in range(k_fold+1):
                if i == num_fold:
                    continue
                image_files += glob.glob(os.path.join(data_root, f"f{i}/*/*"))
            self.image_files = [file[a:] for file in image_files]

        elif data_mode == "val" or data_mode == "test":
            image_files = glob.glob(os.path.join(data_root, f"f{num_fold}/*/*"))
            self.image_files = [file[a:] for file in image_files]
            self.image_files = sorted(self.image_files)
        else:
            raise NotImplementedError
        print(f"{data_mode} dataset fold{num_fold}/{k_fold}: {len(self.image_files)} images")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = self.image_files[idx]

        image_path = os.path.join(self.data_root, file)
        label_path = os.path.join(self.target_root, file)

        image, label = fetch(image_path, label_path)
        image_size = image.size

        if self.data_mode == "train":  # 数据增强
            # image = random_Color(image)
            image = random_Contrast(image, 0.3)
            image = random_Brightness(image, 0.2)
            image = random_GaussianBlur(image, 0.3)

        image, label = convert_to_tensor(image, label, mean, std)

        image = image.repeat((3, 1, 1))   # 将灰度图增加成3通道
        # -------标签处理-------
        label_1 = (label == 255).float()
        label_2 = (label == 190).float()
        label_3 = (label == 105).float()
        label_4 = (label == 0).float()
        label = torch.cat((label_4, label_3, label_2, label_1), dim=0).max(dim=0, keepdim=True)[1].float()
        # -------标签处理-------

        if self.data_mode == "train":  # 数据增强
            image, label = random_Left_Right_filp(image, label)

        label = label.squeeze()

        return {
            "image": image,
            "label": label,
            "file": file,
            "image_size": torch.tensor((image_size[1], image_size[0]))}

    @classmethod
    def recover_image(self, image, image_size, crop_size=None):
        assert len(image.size()) == 3

        # image = F.interpolate(image.unsqueeze(), size=(image_size[0], image_size[1]), mode="bilinear",
        #                       align_corners=True)
        # image = image.squeeze(0)

        return image


class volume_Dataset(Dataset):
    def __init__(self, data_root, target_root, crop_size):
        super(volume_Dataset, self).__init__()
        self.mean = mean
        self.std = std
        self.data_root = data_root
        self.target_root = target_root
        self.crop_size = crop_size

        self.files = sorted(os.listdir(data_root))
        # print(f"pred dataset:{len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        image_path = os.path.join(self.data_root, file)
        label_path = os.path.join(self.target_root, file)

        image, label = fetch(image_path, label_path)
        image_size = image.size
        image, label = convert_to_tensor(image, label, mean, std)
        image = image.repeat((3, 1, 1))   # 将灰度图增加成3通道

        # -------标签处理-------
        label_1 = (label == 255).float()
        label_2 = (label == 190).float()
        label_3 = (label == 105).float()
        label_4 = (label == 0).float()
        label = torch.cat((label_4, label_3, label_2, label_1), dim=0).max(dim=0, keepdim=True)[1].float()
        # -------标签处理-------
        label = label.squeeze()

        return {
            "image": image,
            "label": label,
            "file": file,
            "image_size": torch.tensor((image_size[1], image_size[0]))}