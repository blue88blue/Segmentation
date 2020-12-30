import os
from PIL import Image
import csv
import SimpleITK as sitk
import numpy as np

def resize_data_ISIC(train_image, train_mask, size=(256, 192)):
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



def resize_data_PALM(train_image, train_mask, size=(512, 512)):
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

        #         roi = np.where(roi < 150, roi, 0)
        #         roi_x = np.expand_dims(image.sum(axis=1), 1)
        #         roi_y = np.expand_dims(image.sum(axis=2), 2)
        #         roi = roi_x*roi_y
        #         roi = roi/roi.max()*255

        ###############
        # 标签3D
        volume_file_name = os.path.splitext(volume_list[i])[0]
        if "ct" in volume_file_name:
            volume_file_name = volume_file_name[:-3]
        volume_mask = os.path.join(image_dir, volume_file_name + "_seg.nii")
        if os.path.exists(volume_mask):
            m = sitk.ReadImage(volume_mask)
            print(m.GetPixelIDTypeAsString())
            mask = np.array(sitk.GetArrayViewFromImage(m), np.int8) * 255  # 转换为numpy矩阵 int8
        else:
            mask = None

        # 将每个切片保存
        print(f"\r{i + 1} {image.shape}", end="")

        for num in range(image.shape[0]):
            #             roi_ = Image.fromarray(roi[num, ...]).convert("L")
            #             roi_.save(os.path.join(out_image_dir, volume_file_name+f"_{num}_roi.png"))
            image_ = Image.fromarray(image[num, ...]).convert("L")
            image_.save(os.path.join(out_image_dir, volume_file_name + f"_{num}.png"))
            if mask is not None:
                mask_ = Image.fromarray(mask[num, ...]).convert("L")
                mask_.save(os.path.join(out_mask_dir, volume_file_name + f"_{num}.png"))
