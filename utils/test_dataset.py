from dataset.dataset_PALM import myDataset
from torchvision.utils import save_image
import torch
from settings_PALM import basic_setting
import matplotlib.pyplot as plt
from model.base_models.EfficientNet.model import EfficientNet

if __name__ == "__main__":
    data_root = '/home/sjh/dataset/PLAM/PALM-Training400/PALM-Training400'
    target_root = "/home/sjh/dataset/PLAM/PALM-Training400/PALM-Training400-Annotation-Lession/Lesion_Masks/Atrophy1"  # 萎缩标签
    crop_size = (448, 448)   # (h, w)
    mode = "train"
    dataset = myDataset(data_root, target_root, crop_size, mode)

    batch = dataset[12]
    image = batch["image"]
    label = batch["label"]

    save_image(image, "image.jpg")
    save_image(label, "label.png", normalize=True)

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


