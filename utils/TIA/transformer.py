# -*- coding:utf-8 -*-
from utils.TIA.augment import distort, stretch, perspective
import numpy as np
from PIL import Image

def TIA_trans(img,probs=[0.5,0.5,0.5]):
    # convert PIl image to cv2
    img = np.asarray(img)
    # 训练集中数据像素太低导致进行tia变换时报错，略过像素过低的图像
    if img.shape[:2][1]<20 or img.shape[:2][0]<20:
        return Image.fromarray(img)
    if np.random.binomial(1,probs[0]):
        img = distort(img, 4)
    if np.random.binomial(1,probs[1]):
        img = stretch(img, 4)
    if np.random.binomial(1,probs[2]):
        img = perspective(img)

    # # show imgs
    # import cv2
    # import matplotlib.pyplot as plt
    # pre = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
    # plt.imshow(pre)
    # plt.show()
    # img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img1)
    # plt.show()
    img = Image.fromarray(img)

    return img

if __name__=='__main__':
    TIA_trans('/root/Design/utils/TIA/imgs/word_3919.jpg')