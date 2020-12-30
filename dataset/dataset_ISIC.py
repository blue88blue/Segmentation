from torch.utils.data import Dataset
import csv
import os
from PIL import Image
from .transform import*


mean = torch.Tensor(np.array([0.7085446,  0.58216874, 0.53626412]))
std = torch.Tensor(np.array([0.09625552, 0.11072131, 0.12459033]))


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
                # self.image_files = self.image_files[:200]
            elif data_mode == "val" or data_mode == "test":
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
        label_path = os.path.join(self.target_root, file_name+"_segmentation.png")

        image, label = fetch(image_path, label_path)

        if self.data_mode == "train":  # 数据增强
            image = random_Contrast(image)
            image = random_Brightness(image)

        image, label = convert_to_tensor(image, label, mean=mean, std=std)
        # -------标签处理-------
        label = (label >= 128).float()
        # -------标签处理-------
        # image, label = resize(self.crop_size, image, label)

        if self.data_mode == "train":  # 数据增强
            image, label = random_Top_Bottom_filp(image, label)
            image, label = random_Left_Right_filp(image, label)

        label = label.squeeze()
        return {
            "image": image,
            "label": label,
            "file": file}


def resize_data(train_image, train_mask, size=(256, 192)):
    train_image_resize = train_image + "_resize"
    train_mask_resize = train_mask + "_resize"
    if not os.path.exists(train_image_resize):
        os.mkdir(train_image_resize)
    if not os.path.exists(train_mask_resize):
        os.mkdir(train_mask_resize)

    # 训练集resize
    train_image_list = sorted(os.listdir(train_image))
    for file in train_image_list:
        print(file)
        file_name = os.path.splitext(file)[0]
        image_file = os.path.join(train_image, file)
        mask_file = os.path.join(train_mask, file_name + "_segmentation.png")
        image = Image.open(image_file)
        mask = Image.open(mask_file) if os.path.exists(mask_file) else None

        # if mask is not None:  # 只保存有病的图片
        image = image.resize(size, Image.BILINEAR)
        image.save(os.path.join(train_image_resize, file))
        mask = mask.resize(size, Image.NEAREST)
        mask.save(os.path.join(train_mask_resize, file_name + "_segmentation.png"))

# resize_data('/media/sjh/disk1T/dataset/ISIC/ISIC2018_Task1-2_Training_Input', "/media/sjh/disk1T/dataset/ISIC/ISIC2018_Task1_Training_GroundTruth")