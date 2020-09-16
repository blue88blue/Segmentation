from dataset.dataset_seg import AI_Dataset
from torchvision.utils import save_image
import torch
from settings import basic_setting
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = basic_setting()
    mode = "train"
    dataset = AI_Dataset(args.data_root, args.target_root, args.crop_size, mode)

    batch = dataset[16423]
    image = batch["image"]
    label = batch["label"]

    save_image(image, "image.jpg")
    save_image(label, "label.png", normalize=True)

    print(batch["file"])
    print(image.size())
    print(label.size())
    print(torch.unique(label))



