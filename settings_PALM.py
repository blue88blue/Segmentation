import os
import time
import csv
from dataset.dataset_PALM import *

class basic_setting():

    mode = "test"                             # train,  test,  train_test
    k_fold = 4                              # None 不交叉验证 验证集即为训练集
    start_fold = 0
    end_fold = k_fold

    # #################################### train Data settings ####################################
    dataset_file_list = "utils/PALM_dataset_list.csv"  # 交叉验证所需文件名列表
    data_root = '/home/sjh/dataset/PLAM/PALM-Training400/PALM-Training400'
    target_root = "/home/sjh/dataset/PLAM/PALM-Training400/PALM-Training400-Annotation-Lession/Lesion_Masks/Atrophy1"  # 萎缩标签
    crop_size = (448, 448)

    # #################################### train file settings ####################################
    run_dir = "/media/sjh/disk1T/RUNS/PALM"                      # 数据集名称
    val_step = 2                          # 每训练几个epoch进行一次验证

    # #################################### model settings ####################################
    in_channel = 3
    n_class = 2
    network = "EMUPNet"  # 模型名， 或实验名称
    note = ""  # 标签(区分不同训练设置)
    Ulikenet_channel_reduction = 2  # 类Unet模型通道衰减数(默认通道减半)
    backbone = "resnet34"  # 继承自SegBaseModel的模型backbone
    pretrained = True
    dilated = False
    deep_stem = False
    aux = False

    # #################################### train settings ####################################
    optim = "Adam"
    class_weight = [0.5, 0.5]
    OHEM = False
    num_epochs = 250
    batch_size = 4
    num_workers = 8
    aux_weight = 0.25
    dice_weight = 0.5
    lr = 0.0001
    momentum = 0.9
    weight_decay = 1e-4
    # cuda_id = "0"

    # #################################### test settings ####################################
    test_run_file = "2020-1216-1749_28_EMUPNet__fold_4"
    label_names = ["bg", "atrophy"]
    plot = True  # 保存测试预测图片


    def __init__(self):
        if not os.path.exists("./runs"):
            os.mkdir("./runs")
        # self.run_dir = "./runs/"+self.run_dir
        if not os.path.exists(self.run_dir):
            os.mkdir(self.run_dir)

        if self.mode == "train" or self.mode == "train_test":
            time_now = time.strftime("%Y-%m%d-%H%M_%S", time.localtime(time.time()))
            # 本次训练数据文件夹 模型名+时间
            self.dir = os.path.join(self.run_dir, time_now+"_"+self.network+"_"+self.note + f'_fold_{self.k_fold}')
            os.mkdir(self.dir)

            # 训练验证指标记录文件
            self.val_result_file = os.path.join(self.dir, "val_result.csv")
            with open(self.val_result_file, "a") as f:
                w = csv.writer(f)
                w.writerow(['fold', 'epoch', 'mDice', 'mIoU', 'mAcc', 'mSens', 'mSpec'])

            # 日志保存文件夹
            self.log_dir = os.path.join(self.dir, "log")
            os.mkdir(self.log_dir)
            self.log_dir = [self.log_dir]

            # 模型保存文件夹
            self.checkpoint_dir = os.path.join(self.dir, "checkpoints")
            os.mkdir(self.checkpoint_dir)
            self.checkpoint_dir = [self.checkpoint_dir]

            # 交叉验证的模型保存路径
            if self.k_fold is not None:
                for i in range(self.k_fold):
                    cp_i_dir = os.path.join(self.checkpoint_dir[0], f"fold_{i+1}")
                    log_i_dir = os.path.join(self.log_dir[0], f"fold_{i+1}")
                    os.mkdir(cp_i_dir)
                    os.mkdir(log_i_dir)
                    self.checkpoint_dir.append(cp_i_dir)
                    self.log_dir.append(log_i_dir)
            self.logger(self.dir+"/train_log")

        if self.mode == "test" or self.mode == "train_test":
            if self.mode == "test":
                self.dir = os.path.join(self.run_dir, self.test_run_file)

                # 训练验证指标记录文件, 查询最优模型计算指标
                self.val_result_file = os.path.join(self.dir, "val_result.csv")

                # 模型文件夹
                self.checkpoint_dir = [os.path.join(self.dir, "checkpoints")]
                if self.k_fold is not None:
                    for i in range(self.k_fold):
                        fold_i_dir = os.path.join(self.checkpoint_dir[0], f"fold_{i+1}")
                        self.checkpoint_dir.append(fold_i_dir)
            # 测试结果文件
            self.test_result_file = os.path.join(self.dir, "test_result.csv")
            with open(self.test_result_file, "w") as f:
                w = csv.writer(f)
                title = ['file', 'mDice'] + [name+"_dice" for name in self.label_names[1:]] + \
                        ['mIoU'] + [name + "_iou" for name in self.label_names[1:]] + \
                        ['mAcc'] + \
                        ['mSens'] + [name + "_sens" for name in self.label_names[1:]] + \
                        ['mSpec'] + [name + "_spec" for name in self.label_names[1:]]
                w.writerow(title)
            # 测试结果图片保存
            if self.plot:
                self.plot_save_dir = os.path.join(self.dir, "test_images")
                if not os.path.exists(self.plot_save_dir):
                    os.mkdir(self.plot_save_dir)


    # 记录训练参数与信息
    def logger(self, file):
        with open(file, "a") as f:
            attrs = dir(self)
            for att in attrs:
                if ("__"or "test_" or "val_" or "root" or "logger" or "dir") not in att:
                    f.write(f'{str(att)}:    {str(getattr(self, att))}\n\n')
            f.close()

