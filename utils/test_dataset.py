from dataset.dataset_seg import AI_Dataset
from torchvision.utils import save_image
import torch
from settings import basic_setting
import matplotlib.pyplot as plt
from model.base_models.EfficientNet.model import EfficientNet

if __name__ == "__main__":
    # data_root = "/home/mipav/dataset/AI+/train/image"
    # target_root = '/home/mipav/dataset/AI+/train/label'
    # crop_size = (256, 256)   # (h, w)
    # mode = "train"
    # dataset = AI_Dataset(data_root, target_root, crop_size, mode)
    #
    # batch = dataset[16423]
    # image = batch["image"]
    # label = batch["label"]
    #
    # save_image(image, "image.jpg")
    # save_image(label, "label.png", normalize=True)
    #
    # print(batch["file"])
    # print(image.size())
    # print(label.size())
    # print(torch.unique(label))

    img = torch.randn(2, 3, 224, 224)
    model = EfficientNet.from_pretrained("efficientnet-b4")
    out = model.extract_features_last(img)
    out1 = model.extract_features_last(img)
    features = model.extract_features(img)
    for f in features:
        print(f.size())

    for b in model._blocks_args:
        print(b)
    print(model._global_params)

    print(out1 == out)

