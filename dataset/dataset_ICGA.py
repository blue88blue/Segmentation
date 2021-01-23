from torch.utils.data import Dataset
import csv
import os
from PIL import Image
from torchvision.utils import save_image
from .transform import*


mean = torch.Tensor(np.array([0.37474884]))
std = torch.Tensor(np.array([0.22861719]))


class myDataset(Dataset):
    def __init__(self, data_root, target_root, crop_size, data_mode, k_fold=None, imagefile_csv=None, num_fold=None):
        self.crop_size = crop_size  # h, w
        self.data_root = data_root
        self.target_root = target_root
        self.data_mode = data_mode
        # 若不交叉验证，直接读取data_root下文件列表
        if k_fold == None:
            self.image_files = os.listdir(data_root)
            print(f"{data_mode} dataset: {len(self.image_files)}")
        # 交叉验证：传入包含所有数据集文件名的csv， 根据本次折数num_fold获取文件名列表
        else:
            with open(imagefile_csv, "r") as f:
                reader = csv.reader(f)
                image_files = list(reader)[0]
            fold_size = len(image_files) // k_fold  # 等分
            fold = num_fold - 1
            if data_mode == "train":
                self.image_files = image_files[0: fold*fold_size] + image_files[fold*fold_size+fold_size:]
            elif data_mode == "val" or data_mode == "test":
                if num_fold == k_fold:
                    self.image_files = image_files[fold * fold_size: ]
                else:
                    self.image_files = image_files[fold*fold_size: fold*fold_size+fold_size]
            else:
                raise NotImplementedError
            print(f"{data_mode} dataset fold{num_fold}/{k_fold}: {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = self.image_files[idx]
        file_name, _ = os.path.splitext(file)
        image_path = os.path.join(self.data_root, file)
        label_path = os.path.join(self.target_root, file_name+"_GT.png")

        image, label = fetch(image_path, label_path)
        image_size = image.size

        if self.data_mode == "train":  # 数据增强
            image = random_Contrast(image, 0.3)
            image, label = random_Rotate(image, label)

        image, label = convert_to_tensor(image, label, mean=mean, std=std,norm=False)
        image = image.repeat((3, 1, 1))  # 将灰度图增加成3通道
        # -------标签处理-------
        label = (label == 240).float()
        # -------标签处理-------

        if self.data_mode == "train":  # 数据增强
            image, label = random_Top_Bottom_filp(image, label)
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

        # image = F.interpolate(image.unsqueeze(), size=(image_size[0], image_size[1]), mode="bilinear", align_corners=True)
        # image = image.squeeze(0)

        return image



