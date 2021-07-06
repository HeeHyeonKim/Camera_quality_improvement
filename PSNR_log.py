import os
import cv2
import codes.utils.util as util
from tqdm import tqdm

"""

- Calculate PSNR (GT, LR) to train SR branchs

- Next step : Train MSRResNet and Arrange training set according to the order of PSNR

"""

GT_path = "./datasets/train_GT_sub_img"
LR_path = "./datasets/train_LR_sub_img"

GT_list = os.listdir(GT_path)
LR_list = os.listdir(LR_path)

psnr_log = open("./PSNR_logs.txt", 'w')

for i in tqdm(range(len(GT_list))):

    GT_name = GT_list[i]
    LR_name = LR_list[i]

    GT_img = cv2.imread(os.path.join(GT_path, GT_name), cv2.IMREAD_COLOR)
    LR_img = cv2.imread(os.path.join(LR_path, LR_name), cv2.IMREAD_COLOR)

    psnr = util.calculate_psnr(GT_img, LR_img)
    
    line = "%s, - PSNR : %f \n"%(GT_name, psnr)
    psnr_log.write(line)

psnr_log.close()