
import os.path as osp
import os
import numpy as np
import shutil


#divide training data
LR_folder= './datasets/train_label_img'
GT_folder= './datasets/train_input_img'

save_list=["./datasets/scale_sub_psnr_LR_class3",
           "./datasets/scale_sub_psnr_LR_class2",
           "./datasets/scale_sub_psnr_LR_class1",
           "./datasets/scale_sub_psnr_GT_class3",
           "./datasets/scale_sub_psnr_GT_class2",
           "./datasets/scale_sub_psnr_GT_class1"]
           
for i in save_list:
    if os.path.exists(i):
        pass
    else:
        os.makedirs(i)

threshold=[27.16882,35.149761]

f1 = open("./codes/data_scripts/divide_train.log")
a1 = f1.readlines()

txt_file = open("./datasets/PSNR_log.txt", 'w')

index=0
for i in a1:
    # i = 1 line
    index+=1
    print(index)

    if ('- PSNR:' in i and 'INFO:' in i) and ('results' not in i):
        psnr=float(i.split('PSNR: ')[1].split(' dB')[0])
        filename=i.split('INFO: ')[1].split(' ')[0]
        filename=filename+".png"
        print(filename,psnr)
        log = str(filename) + ", " + str(psnr) + '\n'
        txt_file.write(log)

        # if psnr < threshold[0]:
        #     shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[0], filename))
        #     shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[3], filename))
        # if psnr >= threshold[0] and psnr < threshold[1]:
        #     shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[1], filename))
        #     shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[4], filename))
        # if psnr >= threshold[1]:
        #     shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[2], filename))
        #     shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[5], filename))

txt_file.close()
f1.close()

