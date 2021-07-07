
import os.path as osp
import os
import numpy as np
import shutil
from tqdm import tqdm


#divide training data
LR_folder= '.\\datasets\\train_LR_sub_img'
GT_folder= '.\\datasets\\train_GT_sub_img'

save_list=[".\\datasets\\SR_branches_datasets\\scale_sub_psnr_LR_class3", # Hard
           ".\\datasets\\SR_branches_datasets\\scale_sub_psnr_LR_class2", # Middle
           ".\\datasets\\SR_branches_datasets\\scale_sub_psnr_LR_class1", # Easy
           ".\\datasets\\SR_branches_datasets\\scale_sub_psnr_GT_class3",
           ".\\datasets\\SR_branches_datasets\\scale_sub_psnr_GT_class2",
           ".\\datasets\\SR_branches_datasets\\scale_sub_psnr_GT_class1"]
           
for i in save_list:
    if os.path.exists(i):
        pass
    else:
        os.makedirs(i)

threshold=[20.809031,32.972153]

txt_file = open("./PSNR_logs.txt", 'r')
a1 = txt_file.readlines()

index=0
for i in tqdm(a1):
    # i = 1 line
    index+=1
    # print(index)

    psnr=float(i.split(" : ")[1])
    filename=i.split(",")[0] # GT
    new_filename = filename.split("_")[0] + "_input_" + filename.split("_")[2] + "_" + filename.split("_")[3] # LR
    # print(filename," ",new_filename," ",psnr)

    if psnr < threshold[0]:
        # Hard
        shutil.copy(osp.join(LR_folder, new_filename), osp.join(save_list[0], new_filename))
        shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[3], filename))
    if psnr >= threshold[0] and psnr < threshold[1]:
        # Middle
        shutil.copy(osp.join(LR_folder, new_filename), osp.join(save_list[1], new_filename))
        shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[4], filename))
    if psnr >= threshold[1]:
        # Easy
        shutil.copy(osp.join(LR_folder, new_filename), osp.join(save_list[2], new_filename))
        shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[5], filename))

txt_file.close()
