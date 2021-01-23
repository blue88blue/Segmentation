from torch.utils.data import Dataset
import csv
from .transform import*
import SimpleITK as sitk
import random
import glob


mean = torch.Tensor(np.array([0.22188352]))
std = torch.Tensor(np.array([0.18533521]))



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
            image = random_Contrast(image, 0.3)
            image = random_Brightness(image, 0.2)

        image, label = convert_to_tensor(image, label, mean, std)

        # -------标签处理-------
        label = (label == 255).float()
        # -------标签处理-------

        if self.data_mode == "train":  # 数据增强
            image, label = random_Left_Right_filp(image, label)

        image, label = resize(self.crop_size, image, label)
        label = label.squeeze()

        return {
            "image": image,
            "label": label,
            "file": file,
            "image_size": torch.tensor((image_size[1], image_size[0]))}

    @classmethod
    def recover_image(self, image, image_size, crop_size=None):
        assert len(image.size()) == 3

        image = F.interpolate(image.unsqueeze(), size=(image_size[0], image_size[1]), mode="bilinear",
                              align_corners=True)
        image = image.squeeze(0)

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

        # -------标签处理-------
        label = (label == 255).float()
        # -------标签处理-------
        label = label.squeeze()

        return {
            "image": image,
            "label": label,
            "file": file,
            "image_size": torch.tensor((image_size[1], image_size[0]))}


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





def COVID_19_volume2PNG():
    dir = "/media/sjh/disk1T/dataset/COVID-19-20_v2/Train/"
    output_dir = "/media/sjh/disk1T/dataset/COVID-19-20_v2/train_slices"
    file_list = sorted(os.listdir(dir))
    filename_list = [file[:-10] for file in file_list if "ct" in file]

    for filename in filename_list:
        print(filename)
        volume_path = os.path.join(dir, filename + "_ct.nii.gz")
        volume_gt_path = os.path.join(dir, filename + "_seg.nii.gz")

        volume = sitk.GetArrayFromImage(sitk.ReadImage(volume_path))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(volume_gt_path))

        volume = np.where(volume > (-1400), volume, -1400)
        volume = np.where(volume < 1500, volume, 1500)
        volume = volume + 1400
        volume = (volume / 2900.0) * 255

        out_volume_dir = os.path.join(output_dir + '/img', filename)
        out_gt_dir = os.path.join(output_dir + '/mask', filename)
        os.mkdir(out_volume_dir)
        os.mkdir(out_gt_dir)
        for i in range(volume.shape[0]):
            image = Image.fromarray(volume[i]).convert("L")
            mask = Image.fromarray(gt[i] * 255).convert("L")
            image.save(os.path.join(out_volume_dir, filename + f"_{i}.png"))
            mask.save(os.path.join(out_gt_dir, filename + f"_{i}.png"))

def COVID_19_volume2PNG_2_5D():
    dir = "/media/sjh/disk1T/dataset/COVID-19-20_v2/Train/"
    output_dir = "/media/sjh/disk1T/dataset/COVID-19-20_v2/train_slices_2.5D"
    file_list = sorted(os.listdir(dir))
    filename_list = [file[:-10] for file in file_list if "ct" in file]

    for filename in filename_list:
        print(filename)
        volume_path = os.path.join(dir, filename + "_ct.nii.gz")
        volume_gt_path = os.path.join(dir, filename + "_seg.nii.gz")

        volume = sitk.GetArrayFromImage(sitk.ReadImage(volume_path))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(volume_gt_path))

        volume = np.where(volume > (-1400), volume, -1400)
        volume = np.where(volume < 1500, volume, 1500)
        volume = volume + 1400
        volume = np.array((volume / 2900.0) * 255, np.uint8)
        volume = np.transpose(volume, (1, 2, 0))

        out_volume_dir = os.path.join(output_dir + '/img', filename)
        out_gt_dir = os.path.join(output_dir + '/mask', filename)
        os.mkdir(out_volume_dir)
        os.mkdir(out_gt_dir)
        for i in range(volume.shape[-1]):
            if i == 0 or i == volume.shape[-1] - 1:
                image = np.repeat(np.expand_dims(volume[..., i], -1), 3, axis=-1)
                image = Image.fromarray(image).convert("RGB")
            else:
                image = Image.fromarray(volume[..., i - 1:i + 2]).convert("RGB")
            mask = Image.fromarray(gt[i] * 255).convert("L")
            #         if np.unique(mask) == 0:
            #             if random.random() < 0.1:  # 无病灶的切片保留10%
            #                 image.save(os.path.join(out_volume_dir, filename+f"_{i}.png"))
            #                 mask.save(os.path.join(out_gt_dir, filename+f"_{i}.png"))
            #         else:
            image.save(os.path.join(out_volume_dir, filename + f"_{i}.png"))
            mask.save(os.path.join(out_gt_dir, filename + f"_{i}.png"))



def COVID_19_volume2PNG_2_5D_drop():
    dir = "/media/sjh/disk1T/dataset/COVID-19-20_v2/Train/"
    output_dir = "/media/sjh/disk1T/dataset/COVID-19-20_v2/train_slices_2.5D_drop"
    file_list = sorted(os.listdir(dir))
    filename_list = [file[:-10] for file in file_list if "ct" in file]

    for filename in filename_list:
        print(filename)
        volume_path = os.path.join(dir, filename + "_ct.nii.gz")
        volume_gt_path = os.path.join(dir, filename + "_seg.nii.gz")

        volume = sitk.GetArrayFromImage(sitk.ReadImage(volume_path))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(volume_gt_path))

        volume = np.where(volume > (-1400), volume, -1400)
        volume = np.where(volume < 1500, volume, 1500)
        volume = volume + 1400
        volume = np.array((volume / 2900.0) * 255, np.uint8)
        volume = np.transpose(volume, (1, 2, 0))

        out_volume_dir = os.path.join(output_dir + '/img', filename)
        out_gt_dir = os.path.join(output_dir + '/mask', filename)
        os.mkdir(out_volume_dir)
        os.mkdir(out_gt_dir)
        for i in range(volume.shape[-1]):
            if i == 0 or i == volume.shape[-1] - 1:
                image = np.repeat(np.expand_dims(volume[..., i], -1), 3, axis=-1)
                image = Image.fromarray(image).convert("RGB")
            else:
                image = Image.fromarray(volume[..., i - 1:i + 2]).convert("RGB")
            mask = Image.fromarray(gt[i] * 255).convert("L")
            if np.unique(gt[i]).shape[0] == 1:
                if random.random() < 0.1:  # 无病灶的切片保留10%
                    image.save(os.path.join(out_volume_dir, filename + f"_{i}.png"))
                    mask.save(os.path.join(out_gt_dir, filename + f"_{i}.png"))
            else:
                image.save(os.path.join(out_volume_dir, filename + f"_{i}.png"))
                mask.save(os.path.join(out_gt_dir, filename + f"_{i}.png"))





