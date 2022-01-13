#-*- coding : utf-8-*-
from labelme import utils
import json
import os


def createData(jsonPath):
    try:
        data = json.load(open(jsonPath),)
    except Exception as e:
        print(jsonPath+"  P1")
        print(e)
        return(-1)
    with open(f"./test/test_DB_{data['imagePath'].split()[0]}.txt","w") as f:
        with open("./test.txt","a") as t:
            t.write(f"{jsonPath.split('.')[0]}.jpg ./test/test_DB_{data['imagePath'].split()[0]}.txt\n")
        for shape in data.get("shapes"):
            if len(shape.get("points"))==2:
                x1,y1 = shape.get("points")[0]
                x2,y2 = shape.get("points")[1]
                line = f"{int(x1)},{int(y1)},{int(x2)},{int(y1)},{int(y2)},{int(x1)},{int(x2)},{int(y2)}\n"
                f.write(line)

if __name__=="__main__":
    dir_path = '/root/Design/myProj/actData'
    print(f'command:rm -f ./test/* return:{os.system("rm -f ./test/*")}')
    paths = []
    with open("./test.txt","w") as t:
        pass
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                paths.append(os.path.join(root, file))
    for path in paths:
        createData(path)

