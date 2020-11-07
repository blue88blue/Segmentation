from torch.utils.data import Dataset
import csv
from .transform import*
import SimpleITK as sitk


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
        label_path = os.path.join(self.target_root, file)

        image, label = fetch(image_path, label_path)


        if self.data_mode == "train":  # 数据增强
            # image, label = random_Rotate(image, label)
            image = random_Contrast(image)
            # image = random_Brightness(image)

        image, label = convert_to_tensor(image, label)
        image = image.repeat(3, 1, 1)
        # -------标签处理-------
        label = (label == 255).float()
        # -------标签处理-------

        image, label = resize(self.crop_size, image, label)

        if self.data_mode == "train":  # 数据增强
            image, label = random_Left_Right_filp(image, label)
        label = label.squeeze()

        return {
            "image": image,
            "label": label,
            "file": file}


class predict_Dataset(Dataset):
    def __init__(self, data_root, crop_size):
        super(predict_Dataset, self).__init__()
        self.data_root = data_root
        self.crop_size = crop_size

        self.files = sorted(os.listdir(data_root))
        print(f"pred dataset:{len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name, _ = os.path.splitext(self.files[idx])
        image_path = os.path.join(self.data_root, self.files[idx])

        image, _ = fetch(image_path)
        image_size = (image.size[1], image.size[0])  # h, w
        image, _ = convert_to_tensor(image)
        image = image.repeat(3, 1, 1)
        image, _ = resize(self.crop_size, image)

        return {
            "image": image,
            "file_name": file_name,
            "image_size": torch.tensor(image_size)}





def COVID_19_volume2PNG(image_dir, out_dir):
    volume_list = sorted([file for file in os.listdir(image_dir) if "seg" not in file])  # volume图像列表

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_image_dir = os.path.join(out_dir, "image")  # 输出图片地址
    if not os.path.exists(out_image_dir):
        os.mkdir(out_image_dir)
    out_mask_dir = os.path.join(out_dir, "label")  # 输出标签地址
    if not os.path.exists(out_mask_dir):
        os.mkdir(out_mask_dir)

    for i in range(len(volume_list)):

        # 图像3D
        volume_file = os.path.join(image_dir, volume_list[i])
        img = sitk.ReadImage(volume_file)
        image = sitk.GetArrayViewFromImage(img)  # 转换为numpy矩阵
        ###############
        image = np.where(image > (-1400), image, -1400)
        image = np.where(image < 1500, image, 1500)
        image = image - np.min(image)
        image = (image / 2900.0) * 255
        ###############
        # 标签3D
        volume_file_name = os.path.splitext(volume_list[i])[0]
        if "ct" in volume_file_name:
            volume_file_name = volume_file_name[:-3]
        volume_mask = os.path.join(image_dir, volume_file_name + "_seg.nii")
        if os.path.exists(volume_mask):
            m = sitk.ReadImage(volume_mask)
            mask = np.array(sitk.GetArrayViewFromImage(m), np.int8) * 255  # 转换为numpy矩阵 int8
        else:
            mask = None
        # 将每个切片保存
        print(f"\r{i + 1} {image.shape}", end="")
        for num in range(image.shape[0]):
            image_ = Image.fromarray(image[num, ...]).convert("L")
            image_.save(os.path.join(out_image_dir, volume_file_name + f"_{num}.png"))
            if mask is not None:
                mask_ = Image.fromarray(mask[num, ...]).convert("L")
                mask_.save(os.path.join(out_mask_dir, volume_file_name + f"_{num}.png"))











