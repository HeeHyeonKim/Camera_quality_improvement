"""

train.csv, test.csv 파일을 읽어와
학습에 사용하기 용이하도록 txt 파일을 생성합니다.

"""

import os
import csv
import torch
import numpy as np
import shutil
from tqdm import tqdm

# train_path = "./dataset/train.csv"
# test_path = "./dataset/test.csv"

# train_data_path = "./dataset/train_input_img_256X256"


def csv_to_txt(train_path,test_path):

    train_csv = open(train_path, 'r')
    test_csv = open(test_path, 'r')

    train_txt_path = "./dataset/train.txt"
    test_txt_path = "./dataset/train.txt"

    if not os.path.isdir(train_txt_path):
        os.mkdir(train_txt_path)

    if not os.path.isdir(test_txt_path):
        os.mkdir(test_txt_path)

    train_txt = open("./dataset/train.txt", 'w')
    test_txt = open("./dataset/test.txt", 'w')

    train_reader = csv.reader(train_csv)
    for row in train_reader:
        line = row[0] + ", " + row[1] + ", " + row[2] + "\n"
        train_txt.write(line)

    test_reader = csv.reader(test_csv)
    for row in test_reader:
        line = row[0] + ", " + row[1] + ", " + row[2] + "\n"
        test_txt.write(line)

# To train MSRResNet - split . txt file
def split_dataset_txt(train_data_path, train_size=0.85, random_state=1004):
    data_list = os.listdir(train_data_path)
    data_len = len(data_list)
    
    train_len = int(data_len * train_size)
    validation_len = data_len - train_len

    np.random.seed(random_state)
    shuffled = np.random.permutation(data_len)


    train_txt_path = "./datasets/split_train_set/train.txt"
    validation_txt_path = "./datasets/split_train_set/validation.txt"

    f = open(train_txt_path, 'w')
    print("\n >>> Currently spliting training set.")
    for idx in tqdm(shuffled[:train_len]):
        img_id = data_list[idx].split("_")[2]
        img_input = data_list[idx]
        label_img = data_list[idx][:6] + "label" + data_list[idx][11:]
        
        train_line = "%s, %s, %s \n" % (img_id, img_input, label_img)
        f.write(train_line)
    f.close()

    f = open(validation_txt_path, 'w')
    print("\n >>> Currently spliting validation set.")
    for idx in tqdm(shuffled[train_len:]):
        img_id = data_list[idx].split("_")[2]
        img_input = data_list[idx]
        label_img = data_list[idx][:6] + "label" + data_list[idx][11:]
        
        train_line = "%s, %s, %s \n" % (img_id, img_input, label_img)
        f.write(train_line)
    f.close()

    print("\n Completely split all dataset.")

# To train MSRResNet - split .png file
def split_dataset_txt(train_data_path, txt_path, train_size=0.85, random_state=1004):
    data_list = os.listdir(train_data_path)
    data_len = len(data_list)
    
    train_len = int(data_len * train_size)
    validation_len = data_len - train_len

    np.random.seed(random_state)
    shuffled = np.random.permutation(data_len)


    train_txt_path = txt_path + "_train.txt"
    validation_txt_path = txt_path + "_validation.txt"

    f = open(train_txt_path, 'w')
    print("\n >>> Currently spliting training set.")
    for idx in tqdm(shuffled[:train_len]):
        img_id = data_list[idx].split("_")[2]
        img_input = data_list[idx]
        label_img = data_list[idx][:6] + "input" + data_list[idx][11:]
        
        train_line = "%s, %s, %s \n" % (img_id, img_input, label_img)
        f.write(train_line)
    f.close()

    f = open(validation_txt_path, 'w')
    print("\n >>> Currently spliting validation set.")
    for idx in tqdm(shuffled[train_len:]):
        img_id = data_list[idx].split("_")[2]
        img_input = data_list[idx]
        label_img = data_list[idx][:6] + "input" + data_list[idx][11:]
        
        train_line = "%s, %s, %s \n" % (img_id, img_input, label_img)
        f.write(train_line)
    f.close()

    print("\n Completely split all dataset.")

    
    # 1. SR Branches에 대한 Train, Validation set (.txt) split

    """
    save_list=[".\\datasets\\SR_branches_datasets\\scale_sub_psnr_LR_class3", # Hard
            ".\\datasets\\SR_branches_datasets\\scale_sub_psnr_LR_class2", # Middle
            ".\\datasets\\SR_branches_datasets\\scale_sub_psnr_LR_class1", # Easy
            ".\\datasets\\SR_branches_datasets\\scale_sub_psnr_GT_class3",
            ".\\datasets\\SR_branches_datasets\\scale_sub_psnr_GT_class2",
            ".\\datasets\\SR_branches_datasets\\scale_sub_psnr_GT_class1"]

    for i in range(3):
        path = "./datasets/SR_branches_datasets"
        data_path = save_list[3+i] # .\\datasets\\SR_branches_datasets\\scale_sub_psnr_GT_class3
        # data_name = data_path.split("\\")[3] # scale_sub_psnr_GT_class3

        split_dataset_txt(data_path, data_path)

    """


    # 2. SR Branches 개별 학습을 위한 Train, Validation set (.png) split

    """
    mk_dataset_idx = ["./datasets/SR_branches_datasets/scale_sub_psnr_class1_train.txt",
                "./datasets/SR_branches_datasets/scale_sub_psnr_class1_validation.txt",
                "./datasets/SR_branches_datasets/scale_sub_psnr_class2_train.txt",
                "./datasets/SR_branches_datasets/scale_sub_psnr_class2_validation.txt",
                "./datasets/SR_branches_datasets/scale_sub_psnr_class3_train.txt",
                "./datasets/SR_branches_datasets/scale_sub_psnr_class3_validation.txt"]

    destination = ["./datasets/SR_branches_datasets_for_train/class1_train/GT",
            "./datasets/SR_branches_datasets_for_train/class1_train/LR",
            "./datasets/SR_branches_datasets_for_train/class1_validation/GT",
            "./datasets/SR_branches_datasets_for_train/class1_validation/LR",
            "./datasets/SR_branches_datasets_for_train/class2_train/GT",
            "./datasets/SR_branches_datasets_for_train/class2_train/LR",
            "./datasets/SR_branches_datasets_for_train/class2_validation/GT",
            "./datasets/SR_branches_datasets_for_train/class2_validation/LR",
            "./datasets/SR_branches_datasets_for_train/class3_train/GT",
            "./datasets/SR_branches_datasets_for_train/class3_train/LR",
            "./datasets/SR_branches_datasets_for_train/class3_validation/GT",
            "./datasets/SR_branches_datasets_for_train/class3_validation/LR"]

    # mkdir destination directory
    for dir in destination:
        if os.path.exists(dir):
            pass
        else:
            os.makedirs(dir)

    for i in range(6):
        path = mk_dataset_idx[i]
        f = open(path, 'r')
        lines = f.readlines()

        print("Currently generate datasets : ",mk_dataset_idx[i].split("/")[3])
        for line in tqdm(lines):
            GT_name = line.split(", ")[1]
            LR_name = line.split(", ")[2][:-1]

            GT_path = os.path.join("./datasets/train_GT_sub_img", GT_name)
            LR_path = os.path.join("./datasets/train_LR_sub_img", LR_name)


            shutil.copy(GT_path, destination[2*i])
            shutil.copy(LR_path, destination[2*i+1])
        
    """

    # 3. Pretrain pth (가중치) 파일의 구조를 확인하기 위한 작업

    """
    path = "./FSRCNN_branch3.pth"

    pretrain = torch.load(path)

    for item in pretrain :
        print("Key: ", item, "\t", "Value : ", pretrain[item].shape)
    """

# # 4. ClassSR 학습을 위한 Train, Validation (.png) split

mk_dataset_idx = ["./datasets/ClassSR_datasets_for_train/train.txt",
            "./datasets/ClassSR_datasets_for_train/validation.txt",]

destination = ["./datasets/ClassSR_datasets_for_train/train/GT",
        "./datasets/ClassSR_datasets_for_train/train/LR",
        "./datasets/ClassSR_datasets_for_train/validation/GT",
        "./datasets/ClassSR_datasets_for_train/validation/LR",]

# mkdir destination directory
for dir in destination:
    if os.path.exists(dir):
        pass
    else:
        os.makedirs(dir)

for i in range(2):
    path = mk_dataset_idx[i]
    f = open(path, 'r')
    lines = f.readlines()
    print("Currently generate datasets : ",mk_dataset_idx[i].split("/")[3])
    for line in tqdm(lines):
        GT_name = line.split(", ")[1]
        LR_name = line.split(", ")[2][:-1]
        GT_path = os.path.join("./datasets/train_GT_sub_img", GT_name)
        LR_path = os.path.join("./datasets/train_LR_sub_img", LR_name)
        shutil.copy(GT_path, destination[2*i])
        shutil.copy(LR_path, destination[2*i+1])