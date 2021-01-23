from torch.utils.data import Dataset
import csv
from .transform import*

mean = torch.Tensor(np.array([0.35638645, 0.18319202, 0.11195325]))
std = torch.Tensor(np.array([0.2137684,  0.12645332, 0.07798853]))


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
                    self.image_files = image_files[fold * fold_size:]
                else:
                    self.image_files = image_files[fold * fold_size: fold * fold_size + fold_size]
            else:
                raise NotImplementedError
            print(f"{data_mode} dataset fold{num_fold}/{k_fold}: {len(self.image_files)} images")

        self.weight = torch.FloatTensor([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]]).unsqueeze(0).unsqueeze(0)
        self.weight1 = torch.ones((1, 1, 3, 3))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = self.image_files[idx]
        file_name, _ = os.path.splitext(file)

        image_path = os.path.join(self.data_root, file)
        label_path = os.path.join(self.target_root, file_name+"_seg.png")

        image, label = fetch(image_path, label_path)
        image_size = image.size
        if label == None:
            label = Image.new("L", image.size, 0)

        if self.data_mode == "train":  # 数据增强
            image, label = random_transfrom(image, label)

        image, label = convert_to_tensor(image, label, mean, std, norm=True)

        # -------标签处理-------
        label = (label == 255).float()
        # -------标签处理-------

        if self.data_mode == "train":  # 数据增强
            image, label = random_Top_Bottom_filp(image, label)
            image, label = random_Left_Right_filp(image, label)

        label = label.squeeze()

        # df = torch.tensor(direct_field(label.numpy()))   # （2, h, w）
        # df1 = df[0, ...].clone()
        # df2 = df[1, ...].clone()
        # df = torch.cat((df2.unsqueeze(0), df1.unsqueeze(0)), dim=0).contiguous()
        #
        # edge_label = F.conv2d(label.unsqueeze(0).unsqueeze(0), self.weight, padding=1)
        # edge_label = (edge_label > 0).float()
        # for i in range(5):
        #     edge_label = F.conv2d(edge_label, self.weight1, padding=1)
        #     edge_label = (edge_label > 0).float()
        # edge_label = edge_label.squeeze()
        return {
            "image": image,
            "label": label,
            # "df": df,
            # "edge_label": edge_label,
            "file": file,
            "image_size": torch.tensor((image_size[1], image_size[0]))}

    @classmethod
    def recover_image(self, image, image_size, crop_size=None):
        assert len(image.size()) == 3

        return image





class predict_Dataset(Dataset):
    def __init__(self, data_root, crop_size):
        super(predict_Dataset, self).__init__()
        self.mean = mean
        self.std = std
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
        image_size = image.size  # w,h

        image = image.resize((self.crop_size[1], self.crop_size[0]),  Image.BILINEAR)
        image, _ = convert_to_tensor(image, mean=mean, std=std)
        # image, _ = scale_adaptive(self.crop_size, image)
        # image, _ = pad(self.crop_size, image)


        return {
            "image": image,
            "file_name": file_name,
            "image_size": torch.tensor((image_size[1], image_size[0]))}













def resize_data(train_image, train_mask, size=(512, 512)):
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
        mask_file = os.path.join(train_mask, file_name + ".bmp")
        image = Image.open(image_file)
        mask = Image.open(mask_file) if os.path.exists(mask_file) else None

        # if mask is not None:  # 只保存有病的图片
        image = image.resize(size, Image.BILINEAR)
        image.save(os.path.join(train_image_resize, file))
        if mask is not None:
            mask = mask.resize(size, Image.NEAREST)
            mask.save(os.path.join(train_mask_resize, file_name + ".bmp"))





