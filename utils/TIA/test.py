# -*- coding:utf-8 -*-
import cv2
from augment import distort, stretch, perspective
import matplotlib.pyplot as plt
import numpy as np


def TIA_trans(path,probs=[0,0.9,0]):
    img = cv2.imread(path)
    pre = img
    # if np.random.binomial(1,probs[0]):
    #     img = distort(img, 4)
    if np.random.binomial(1,probs[1]):
        img = stretch(img, 1)
    # if np.random.binomial(1,probs[2]):
    #     img = perspective(img)
    
    
    # show imgs
    pre = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
    plt.imshow(pre)
    plt.show()
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.show()

if __name__=='__main__':
    TIA_trans('/root/Design/utils/TIA/imgs/word_3919.jpg')