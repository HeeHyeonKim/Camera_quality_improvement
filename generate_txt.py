"""

train.csv, test.csv 파일을 읽어와
학습에 사용하기 용이하도록 txt 파일을 생성합니다.

"""

import os
import csv
import numpy as np
from tqdm import tqdm

train_path = "./dataset/train.csv"
test_path = "./dataset/test.csv"

train_data_path = "./dataset/train_input_img_256X256"


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

def split_dataset(train_data_path, train_size=0.85, random_state=1004):
    data_list = os.listdir(train_data_path)
    data_len = len(data_list)
    
    train_len = int(data_len * train_size)
    validation_len = data_len - train_len

    np.random.seed(random_state)
    shuffled = np.random.permutation(data_len)


    train_txt_path = "./dataset/train.txt"
    validation_txt_path = "./dataset/validation.txt"

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

split_dataset(train_data_path)