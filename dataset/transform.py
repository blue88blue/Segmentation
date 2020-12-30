import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw, PIL.ImageFilter
import os
Image.MAX_IMAGE_PIXELS = None
import cv2
# 随机旋转
def random_Rotate(img, label=None):
    rand = int(float(torch.rand(1)-0.5)*60)
    img = img.rotate(rand)
    if label is not None:
        label = label.rotate(rand)
    return img, label

# 随机对比度
def random_Contrast(img):
    v = float(torch.rand(1)) * 2
    if 0.5 <= v <= 1.5:
        return PIL.ImageEnhance.Contrast(img).enhance(v)
    else:
        return img

# 随机颜色鲜艳或灰暗
def random_Color(img):
    v = float(torch.rand(1)) * 2
    if 0.4 <= v <= 1.5:
        return PIL.ImageEnhance.Color(img).enhance(v)
    else:
        return img

# 随机亮度变换
def random_Brightness(img):  # [0.1,1.9]
    v = float(torch.rand(1)) * 2
    if 0.4 <= v <= 1.5:
        return PIL.ImageEnhance.Brightness(img).enhance(v)
    else:
        return img

# 随机高斯模糊
def random_GaussianBlur(img):
    p = float(torch.rand(1))
    if p > 0.6:
        v = float(torch.rand(1))+1.2
        return img.filter(PIL.ImageFilter.GaussianBlur(radius=v))
    else:
        return img

# 随机变换
def random_transfrom(image, label=None):
    # image = random_Color(image)
    image, label = random_Rotate(image, label)
    image = random_Contrast(image)
    image = random_Brightness(image)
    # image = random_GaussianBlur(image)
    return image, label


# 读取图片与mask，返回4D张量
def fetch(image_path, label_path=None):
    image = Image.open(image_path)

    if label_path is not None:
        if os.path.exists(label_path):
            label = Image.open(label_path)
        else:
            label = None
    else:
        label = None

    return image, label

# image转为tensor
def convert_to_tensor(image, label=None, mean=0.5, std=0.5):
    image = torch.FloatTensor(np.array(image)) / 255
    image = (image - mean) / std
    # image = image - torch.Tensor(np.array([0.5, 0.5, 0.5]))
    if len(image.size()) == 2:
        image = image.unsqueeze(0)
    else:
        image = image.permute(2, 0, 1)

    if label is not None:
        label = torch.FloatTensor(np.array(label))
        if len(label.size()) == 2:
            label = label.unsqueeze(0)
    else:
        label = torch.zeros((1, image.size()[1], image.size()[2]))
    return image, label

# 根据比例resize
def scale(crop_ratio, image, label=None):
    size_h, size_w = image.size()[-2:]
    size = (int(crop_ratio*size_h), int(crop_ratio*size_w))

    image = F.interpolate(image.unsqueeze(0), size = size, mode='bilinear', align_corners=True).squeeze(0)
    if label is not None:
        label = F.interpolate(label.unsqueeze(0), size = size, mode='nearest').squeeze(0)
    return image, label


def scale_adaptive(crop_size, image, label=None):
    image_size = image.size()[-2:]
    ratio_h = float(crop_size[0] / image_size[0])
    ratio_w = float(crop_size[1] / image_size[1])
    ratio = min(ratio_h, ratio_w)
    size = (int(image_size[0]*ratio), int(image_size[1]*ratio))

    image = F.interpolate(image.unsqueeze(0), size=size, mode='bilinear', align_corners=True).squeeze(0)
    if label is not None:
        label = F.interpolate(label.unsqueeze(0), size=size, mode='nearest').squeeze(0)
    return image, label




# resize
def resize(crop_size, image, label=None):
    image = F.interpolate(image.unsqueeze(0), size=crop_size, mode='bilinear', align_corners=True).squeeze(0)
    if label is not None:
        label = F.interpolate(label.unsqueeze(0), size=crop_size, mode='nearest').squeeze(0)
    return image, label 


# 随机裁剪
def random_crop(crop_size, image, label=None):
    assert len(image.size()) == 3
    h, w = image.size()[-2:]
    delta_h = h-crop_size[0]
    delta_w = w-crop_size[1]

    sh = int(torch.rand(1)*delta_h) if delta_h > 0 else 0
    sw = int(torch.rand(1)*delta_w) if delta_w > 0 else 0
    eh = crop_size[0]+sh if delta_h > 0 else h
    ew = crop_size[1]+sw if delta_w > 0 else w
    
    image = image[:, sh:eh, sw:ew]
    if label is not None:
        assert len(label.size()) == 3
        label = label[:, sh:eh, sw:ew]

    return image, label


# 若原图小于裁剪图，填充
def pad(crop_size, image, label=None, pad_value=0.0):
    h, w = image.size()[-2:]
    pad_h = max(crop_size[0] - h, 0)
    pad_w = max(crop_size[1] - w, 0)
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=image.min())
        if label is not None:
            label = F.pad(label, (0, pad_w, 0, pad_h), mode='constant', value=pad_value)
    return image, label


# 随机垂直翻转
def random_Top_Bottom_filp(image, label=None, p=0.5):
    a = float(torch.rand(1))
    if a > p:
        image = torch.flip(image, [1])
        if label is not None:
            if len(label.size()) == 2:
                label = label.unsqueeze(0)
            label = torch.flip(label, [1])
    return image, label


# 随机垂直翻转
def random_Left_Right_filp(image, label=None, p=0.5):
    a = float(torch.rand(1))
    if a > p:
        image = torch.flip(image, [2])
        if label is not None:
            if len(label.size()) == 2:
                label = label.unsqueeze(0)
            label = torch.flip(label, [2])
    return image, label


def direct_field(a, norm=True):
    """ a: np.ndarray, (h, w)
    """
    if a.ndim == 3:
        a = np.squeeze(a)

    h, w = a.shape

    a_Image = Image.fromarray(a)
    a = a_Image.resize((w, h), Image.NEAREST)
    a = np.array(a)

    accumulation = np.zeros((2, h, w), dtype=np.float32)
    for i in np.unique(a)[1:]:
        # b, ind = ndimage.distance_transform_edt(a==i, return_indices=True)
        # c = np.indices((h, w))
        # diff = c - ind
        # dr = np.sqrt(np.sum(diff ** 2, axis=0))

        img = (a == i).astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                                                      labelType=cv2.DIST_LABEL_PIXEL)
        index = np.copy(labels)
        index[img > 0] = 0
        place = np.argwhere(index > 0)
        nearCord = place[labels - 1, :]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, h, w))
        nearPixel[0, :, :] = x
        nearPixel[1, :, :] = y
        grid = np.indices(img.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel
        if norm:
            dr = np.sqrt(np.sum(diff ** 2, axis=0))
        else:
            dr = np.ones_like(img)

        # direction = np.zeros((2, h, w), dtype=np.float32)
        # direction[0, b>0] = np.divide(diff[0, b>0], dr[b>0])
        # direction[1, b>0] = np.divide(diff[1, b>0], dr[b>0])

        direction = np.zeros((2, h, w), dtype=np.float32)
        direction[0, img > 0] = np.divide(diff[0, img > 0], dr[img > 0])
        direction[1, img > 0] = np.divide(diff[1, img > 0], dr[img > 0])

        accumulation[:, img > 0] = 0
        accumulation = accumulation + direction

    # mag, angle = cv2.cartToPolar(accumulation[0, ...], accumulation[1, ...])
    # for l in np.unique(a)[1:]:
    #     mag_i = mag[a==l].astype(float)
    #     t = 1 / mag_i * mag_i.max()
    #     mag[a==l] = t
    # x, y = cv2.polarToCart(mag, angle)
    # accumulation = np.stack([x, y], axis=0)

    return accumulation









