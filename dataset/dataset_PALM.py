from torch.utils.data import Dataset
import csv
from .transform import*


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
                self.image_files = self.image_files * 2   # ################ *2
            elif data_mode == "val" or data_mode == "test":
                self.image_files = image_files[fold*fold_size: fold*fold_size+fold_size]
            else:
                raise NotImplementedError
            print(f"{data_mode} dataset fold{num_fold}/{k_fold}: {len(self.image_files)} images")

        # self.weight = torch.FloatTensor([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]]).unsqueeze(0).unsqueeze(0)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = self.image_files[idx]
        file_name, _ = os.path.splitext(file)

        image_path = os.path.join(self.data_root, file)
        label_path = os.path.join(self.target_root, file_name+".bmp")

        image, label = fetch(image_path, label_path)
        if label == None:
            label = Image.new("L", image.size, 0)

        if self.data_mode == "train":  # 数据增强
            image, label = random_transfrom(image, label)

        image, label = convert_to_tensor(image, label)

        # -------标签处理-------
        label = (label == 255).float()
        # -------标签处理-------

        image, label = scale_adaptive(self.crop_size, image, label)

        if self.data_mode == "train":  # 数据增强
            image, label = random_Top_Bottom_filp(image, label)
            image, label = random_Left_Right_filp(image, label)

        image, label = pad(self.crop_size, image, label)
        label = label.squeeze()

        # edge_label = F.conv2d(label.unsqueeze(0).unsqueeze(0), self.weight, padding=1)
        # edge_label = (edge_label.squeeze() > 0).float()
        return {
            "image": image,
            "label": label,
            # "edge_label": edge_label,
            "file": file}


class predict_Dataset(Dataset):
    def __init__(self, data_root, crop_size):
        super(predict_Dataset, self).__init__()
        self.data_root = data_root
        self.crop_size = crop_size

        self.files = os.listdir(data_root)
        print(f"pred dataset:{len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name, _ = os.path.splitext(self.files[idx])
        image_path = os.path.join(self.data_root, self.files[idx])

        image, _ = fetch(image_path)
        image_size = image.size  # w,h
        image, _ = convert_to_tensor(image)
        image, _ = scale_adaptive(self.crop_size, image)
        image, _ = pad(self.crop_size, image)

        return {
            "image": image,
            "file_name": file_name}
