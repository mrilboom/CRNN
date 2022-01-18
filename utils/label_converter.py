#使用的数据集并不是  图像——中文  的格式，而是  图像——数字标签  的格式，
#需要将该数字标签按照alphabet转换为中文

import alphabets

A = alphabets.alphabet

with open("/data/zjj/origin/train2.txt","w") as f2:
    pass
with open("/data/zjj/origin/train.txt","r") as f1:
    with open("/data/zjj/origin/train2.txt","a") as f2:
        for line in f1.readlines():
            imgpath = line.split(" ")[0]
            labels = line.split(" ")[1:]
            f2.write(imgpath+" ")
            for label in labels:
                label.replace("\n","")
                f2.write(A[int(label)-1])
            f2.write("\n")
            # print(1)