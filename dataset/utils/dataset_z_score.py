import os
from PIL import Image
import csv
import numpy as np

def get_mean_var(image_dir, list_name, ignore_value=-1, channel=3):
    image_list = sorted(os.listdir(image_dir))
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




get_mean_var('/home/sjh/dataset/REFUGE2_ROI_512/MICCAI2018/1200_image', "REFUGE_MV_ROI.csv")



















