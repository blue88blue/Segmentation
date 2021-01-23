from dataset.dataset_RETOUCH import myDataset
from torchvision.utils import save_image
import torch
from settings_PALM import basic_setting
import matplotlib.pyplot as plt
from model.base_models.EfficientNet.model import EfficientNet

if __name__ == "__main__":
    data_root = '/media/sjh/disk1T/dataset/RETOUCH_crop/train_all/img'
    target_root = "/media/sjh/disk1T/dataset/RETOUCH_crop/train_all/mask"  # 萎缩标签
    crop_size = (448, 448)   # (h, w)
    mode = "train"
    dataset = myDataset(data_root, target_root, crop_size, mode, k_fold=3, imagefile_csv=None, num_fold=3)

    batch = dataset[45]
    image = batch["image"]
    label = batch["label"]
    # edge_label = batch["edge_label"]

    save_image(image*0.1052+0.2294587, "image.jpg")
    save_image(label/4, "label.png")
    # save_image(edge_label, "edge_label.png", normalize=True)

    print(batch["file"])
    print(image.size())
    print(label.size())
    print(torch.unique(label))

    # img = torch.randn(2, 3, 224, 224)
    # # model = EfficientNet.from_name("efficientnet-b3")
    # model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
    # out = model(img)
    # print(out.size())

    # for f in features:
    #     print(f.size())
    # for b in model._blocks_args:
    #     print(b)
    # print(model._global_params)


