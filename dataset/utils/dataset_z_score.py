import os
from PIL import Image
import csv
import numpy as np
import glob

def get_mean_var(image_dir, list_name, ignore_value=-1, channel=3):
    image_list = sorted(os.listdir(image_dir))
    image_list = [file for file in image_list if "seg" not in file]
    mv_list = ['imgName']+['Mean']*channel+['Var']*channel
    M = []
    V = []
    for i, f in enumerate(image_list):
        print("\r", i, f, end="")
        file = os.path.join(image_dir, f)
        image = np.array(Image.open(file)).reshape(-1, channel)/255
        mask = (image != ignore_value)
        mean = []
        var = []
        for c in range(channel):
            image_vaild = image[mask[:, c], c]
            mean.append(np.mean(image_vaild))
            var.append(np.std(image_vaild))
            # var1 = np.sqrt(np.mean(np.power(image-mean, 2)))
        mv_list.append([f]+mean+var)
        M.append(mean)
        V.append(var)

    m = np.mean(np.array(M), axis=0)
    v = np.mean(np.array(V), axis=0)
    print("Mean", m)
    print("Var", v)
    with open(list_name, "w") as f:
        w = csv.writer(f)
        w.writerows(mv_list)
        w.writerow(['m']+list(m))
        w.writerow(['v']+list(v))

def get_mean_var1(image_dir, list_name, ignore_value=-1, channel=1):
    image_list = sorted(glob.glob(image_dir+"/*/*/*"))
    mv_list = ['imgName']+['Mean']*channel+['Var']*channel
    M = []
    V = []
    for i, f in enumerate(image_list):
        print("\r", i, f, end="")
        image = np.array(Image.open(f)).reshape(-1, channel)/255
        mask = (image != ignore_value)
        mean = []
        var = []
        for c in range(channel):
            image_vaild = image[mask[:, c], c]
            mean.append(np.mean(image_vaild))
            var.append(np.std(image_vaild))
        mv_list.append([f]+mean+var)
        M.append(mean)
        V.append(var)

    m = np.mean(np.array(M), axis=0)
    v = np.mean(np.array(V), axis=0)
    print("Mean", m)
    print("Var", v)
    with open(list_name, "w") as f:
        w = csv.writer(f)
        w.writerows(mv_list)
        w.writerow(['m']+list(m))
        w.writerow(['v']+list(v))

get_mean_var("/home/sjh/dataset/PLAM/data620/crop_512", "PM_data620_mv.csv", channel=3)



# 获取每个mask的目标的面积
def get_mask_S(mask_dir, list_name):
    mask_list = sorted(os.listdir(mask_dir))

    S = []
    for i, f in enumerate(mask_list):
        print("\r", i, f, end="")
        file = os.path.join(mask_dir, f)
        image = np.array(Image.open(file)).squeeze()/255
        S.append([f, np.sum(image)/(image.shape[0]*image.shape[1])])

    with open(list_name, "w") as f:
        w = csv.writer(f)
        w.writerows(S)

# get_mask_S("/home/sjh/dataset/PLAM/PALM-Training400/PALM-Training400-Annotation-Lession/Lesion_Masks/Atrophy1", "PALM_mask_S.csv")














