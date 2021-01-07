import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import pandas as pd
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import collections
from shutil import copyfile


def rle2mask(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo: hi] = 1
    return img.reshape(shape).T


def read_image(base_path, image_id, df_train=None, scale=None, verbose=1):
    image = tifffile.imread(
        os.path.join(base_path, f"{image_id}.tiff")
    )
    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)

    mask = None
    if df_train is not None:
        mask = rle2mask(
            df_train[df_train["id"] == image_id]["encoding"].values[0],
            (image.shape[1], image.shape[0])
        )

    if verbose:
        print(f"[{image_id}] Image shape: {image.shape}")
        if mask is not None:
            print(f"[{image_id}] Mask shape: {mask.shape}")

    if scale:
        new_size = (image.shape[1] // scale, image.shape[0] // scale)
        image = cv2.resize(image, new_size)
        if mask is not None:
            mask = cv2.resize(mask, new_size)

        if verbose:
            print(f"[{image_id}] Resized Image shape: {image.shape}")
            if mask is not None:
                print(f"[{image_id}] Resized Mask shape: {mask.shape}")

    return image, mask


def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs


class crop_image():
    def __init__(self, df_train, src_path="/media/sjh/disk1T/dataset/kidney", crop_size=[256, 256], scale=4):
        self.scale = scale
        self.df_train = df_train
        self.src_path = src_path
        self.crop_size = crop_size
        self.crop_gap = list(np.array(crop_size))

        self.train_ids = [
            "0486052bb", "095bf7a1f", "1e2425f28", "2f6ecfcdf",
            "54f2eec69", "aaa6a05cc", "cb2d976f4", "e79de561c", ]
        self.test_ids = [
            "26dc41664", "afa5e8098", "b2dc8411c", "b9a3865fc",
            "c68fe75ea", ]
        self.ids = [self.train_ids, self.test_ids]

    def mkdir(self, mode, type=0):
        assert mode in ["resize", "crop"]

        if type == 0:
            tiff_path = os.path.join(self.src_path, "train")
            image_path = self.src_path + "/train_image_" + mode
            mask_path = self.src_path + "/train_mask_" + mode
            if not os.path.exists(mask_path):
                os.mkdir(mask_path)
        else:
            tiff_path = os.path.join(self.src_path, "test")
            image_path = self.src_path + "/test_image_" + mode
            mask_path = None

        if not os.path.exists(image_path):
            os.mkdir(image_path)

        return tiff_path, image_path, mask_path

    def get_positions(self, h, w):
        crop_positions = []
        position = [0, self.crop_size[0], 0, self.crop_size[1]]
        while (1):
            if position[1] > h:  # or position[1] > h * 3 / 4:
                position[1] = h
                position[0] = h - self.crop_size[0]
            position[2] = 0
            position[3] = self.crop_size[1]
            while (1):
                if position[3] > w:  # or position[3] > w * 3 / 4:
                    position[3] = w
                    position[2] = w - self.crop_size[1]
                # 记录所有裁剪位置
                crop_positions.append(position.copy())

                if position[3] == w:
                    break
                position[2] += self.crop_gap[1]
                position[3] += self.crop_gap[1]

            if position[1] == h:
                break
            position[0] += self.crop_gap[0]
            position[1] += self.crop_gap[0]

        return crop_positions

    def save_crop_image_mask(self, type=0):
        ids = self.ids[type]
        path, image_path, mask_path = self.mkdir("crop", type)
        df = self.df_train if type == 0 else None
        for image_id in ids:
            image, mask = read_image(path, image_id, df, scale=self.scale, verbose=1)
            if mask is not None:
                mask = mask * 255
            crop_positions = self.get_positions(image.shape[0], image.shape[1])
            for i, position in enumerate(crop_positions):
                crop_image = Image.fromarray(image[position[0]:position[1], position[2]:position[3], :])
                crop_image.save(os.path.join(image_path, f"{image_id}_{i}.png"))
                if mask is not None:
                    crop_mask = Image.fromarray(mask[position[0]:position[1], position[2]:position[3]])
                    crop_mask.save(os.path.join(mask_path, f"{image_id}_{i}.png"))

    def save_resize_image_mask(self, type=0):
        ids = self.ids[type]
        df = self.df_train if type == 0 else None
        path, image_path, mask_path = self.mkdir("resize", type)
        for image_id in ids:
            image, mask = read_image(path, image_id, df, scale=self.scale, verbose=1)
            crop_image = Image.fromarray(image)
            crop_image.save(os.path.join(image_path, f"{image_id}.png"))
            if mask is not None:
                mask = mask * 255
                crop_mask = Image.fromarray(mask)
                crop_mask.save(os.path.join(mask_path, f"{image_id}.png"))

    def merge_score(self, score_dir, image_dir):
        merge_score_dir = score_dir+"_result"
        if not os.path.exists(merge_score_dir):
            os.mkdir(merge_score_dir)

        for image_id in self.test_ids:
            print(image_id)
            image_w, image_h = Image.open(os.path.join(image_dir, image_id+".png")).size
            score_map = np.zeros((image_h, image_w, 2), dtype= np.int16)
            crop_positions = self.get_positions(image_h, image_w)

            for i, position in enumerate(crop_positions):
                crop_score_map = np.array(Image.open(os.path.join(score_dir, image_id + f"_{i}.png")))
                crop_score_map = np.expand_dims(crop_score_map, axis=2)
                crop_score_map_1 = crop_score_map
                crop_score_map_0 = 255 - crop_score_map
                crop_score_map = np.concatenate((crop_score_map_0, crop_score_map_1), axis=2)
                score_map[position[0]:position[1], position[2]:position[3], :] += crop_score_map
            mask = np.argmax(score_map, axis=2)
            mask = Image.fromarray(np.uint8(mask)*255).convert("L")
            mask.save(os.path.join(merge_score_dir, image_id+"_mask.png"))



if __name__ == "__main__":
    BASE_PATH = ""
    df_train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
    m = crop_image(df_train, BASE_PATH)
    # m.save_resize_image_mask(type=0)   # 训练集
    # m.save_resize_image_mask(type=1)   # 测试集
    m.save_crop_image_mask(type=1)












