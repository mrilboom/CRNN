#-*- coding : utf-8-*-
# 读取json文件，转换为标准格式

from labelme import utils
import PIL
from PIL import Image
import json
import os
import albumentations as A
import cv2


def createData(jsonPath):
    transform = A.Compose([
        A.Perspective (scale=(0.05, 0.1), keep_size=True, pad_mode=cv2.BORDER_REPLICATE, pad_val=cv2.BORDER_REPLICATE, mask_pad_val=0, fit_output=True, interpolation=1, always_apply=True, p=0.5),
        A.GaussianBlur(blur_limit=(1, 5), sigma_limit=0, always_apply=False, p=0.5)
    ])
    try:
        data = json.load(open(jsonPath),)
    except Exception as e:
        print(jsonPath+"  P1")
        print(e)
        return(-1)
    imageData = data.get("imageData")
    path = data.get("imagePath").split(".")[0]
    img = utils.img_b64_to_arr(imageData)
    count = 0
    for shape in data.get("shapes"):
        try:
            if len(shape.get("points"))<2:
                continue
            img_c = Image.fromarray(img).crop(tuple(shape.get("points")[0]+shape.get("points")[1]))
            if img_c.width<80:
                continue
            label = shape.get("label")
            if "#" in label:
                continue
            if "机打发票" in label:
                continue
            if "发票联" in label:
                continue
            count+=1
            img_c.save(f"/data/zjj/loc/unpacked/train/{path}_{count}.jpg")
            label = label.replace("：", ":")
            label = label.replace("，", ",")
            with open("label.txt","a") as f:
                f.write(f"{path}_{count}.jpg {label}\n")
            # image = cv2.imread(f"/root/Design/myProj/T_Data/{path}_{count}.jpg")
            # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # for i in range(2):
            #     transformed = transform(image=image)
            #     transformed_image = transformed["image"]
            #     cv2.imwrite(f"/root/Design/myProj/T_Data/{path}_{count}_en{i}.jpg", transformed_image)
            #     with open("label.txt","a") as f:
            #         f.write(f"{path}_{count}_en{i}.jpg {label}\n")

        except Exception as e:
            print(jsonPath+"  P2")
            print(e)
            return(-1)
if __name__=="__main__":
    with open("label.txt","w") as f:
        pass
    dir_path = 'dataset/origin/train/'
    paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                paths.append(os.path.join(root, file))
    for path in paths:
        createData(path)

