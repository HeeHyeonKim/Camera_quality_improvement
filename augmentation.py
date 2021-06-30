import os
import cv2
import numpy as np
from tqdm import tqdm

path_dir = ".\\dataset"


"""
1. count_png_type 함수는 .png 파일의 크기를 확인하고, 크기별 개수를 확인하는 기능을 수행합니다.

    - Current image size : (1224, 1632, 3) : 700개, (2448, 3264, 3) : 564개
"""

def count_png_type(path_dir):

    # dataset 디렉토리에 접근하여 상대경로를 dataset_dirs에 append 합니다.
    dataset_dirs = []
    for name in os.listdir(path_dir):
        path = os.path.join(path_dir,name)
        if os.path.isdir(path):
            dataset_dirs.append(path)

    # dataset 내부 디렉토리에 각각 접근하여 .png 파일들의 크기를 파악합니다.
    shape_list = []
    for directory in dataset_dirs:
        print("\n >>> Currently working directory is < %s >" %directory)
        for file_name in tqdm(os.listdir(directory)):
            img_path = os.path.join(directory,file_name)
            img = cv2.imread(img_path)
            shape_list.append(img.shape)

    # print(set(shape_list))
    type_1 = shape_list.count((1224, 1632, 3))
    type_2 = shape_list.count((2448, 3264, 3))

    print("\n Count (1224, 1632, 3) : ", type_1)
    #  (1224, 1632, 3) :  700 / 2 = 350

    print("\n Count (2448, 3264, 3) : ", type_2)
    #  (2448, 3264, 3) :  564 / 2 = 282



"""
2. cut_img 함수는 데이콘 코드 공유 부분을 수정하여 작성하였다.
    위 함수는 img_path_list 내부의 이미지에 접근하여 매개변수로 받은 stride만큼 슬라이딩하며 
    stride X stride 크기의 데이터 셋(.npy)을 새롭게 생성한다.

    전처리된 이미지는 {path}_256X256 디렉토리에 저장된다.
"""

def cut_img(img_path_list, stride):
    os.makedirs(f'{img_path_list}_256X256', exist_ok=True)
    print("\n >>> Currently working directory is < %s >" %img_path_list)
    for img_name in tqdm(os.listdir(img_path_list)):
        num = 0
        img_path = os.path.join(img_path_list,img_name)
        img_name = img_name[:-4]
        save_path = os.path.join(f'{img_path_list}_256X256',img_name)
        img = cv2.imread(img_path)
        for top in range(0, img.shape[0], stride):
            for left in range(0, img.shape[1], stride):
                piece = np.zeros([256, 256, 3], np.uint8)
                temp = img[top:top+256, left:left+256, :]
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                np.save(f'{save_path}_{num}.npy', piece)
                num+=1


count_png_type(path_dir)

for dir_name in os.listdir(path_dir):
    dir_path = os.path.join(path_dir,dir_name)
    if os.path.isdir(dir_path):
            cut_img(dir_path, 256)