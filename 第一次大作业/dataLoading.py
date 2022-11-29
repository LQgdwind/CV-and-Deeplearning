import json
from pandas import json_normalize
import shutil
import os
import torchvision

def copyfile(filename, target_dir):
    """将⽂件复制到⽬标⽬录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def data_to_train():
    # 读入json中的数据
    with open("data/train.json", "r", encoding='utf-8') as p:
        dataframe = json.load(p)
    dataframe = json_normalize(dataframe)
    path_set = list(dataframe['path'])
    origin_path = []
    target_path = []
    # 把文件路径改成我的目录结构下的路径,并且把训练集给复制过去
    tuple_tmp = [0, 1]
    for i, item in enumerate(path_set):
        word_group = item.split('\\')
        if word_group[1] != tuple_tmp[0]:
            tuple_tmp[1] = 1
            tuple_tmp[0] = word_group[1]
        origin_path.append("data" + "\\" + word_group[1] + "\\" + word_group[2])
        target_path.append("train" + "\\" + word_group[1])
        copyfile(origin_path[i], target_path[i])
        # 重命名为  类别_序号  的风格
        os.rename(target_path[i]+"\\"+word_group[2],target_path[i]+"\\"+word_group[1]+"_"+str(tuple_tmp[1])+".jpg")
        tuple_tmp[1] = tuple_tmp[1]+1

def data_to_valid():
    # 读入json中的数据
    with open("data/val.json", "r", encoding='utf-8') as p:
        dataframe = json.load(p)
    dataframe = json_normalize(dataframe)
    path_set = list(dataframe['path'])
    origin_path = []
    target_path = []
    # 把文件路径改成我的目录结构下的路径,并且把训练集给复制过去
    tuple_tmp = [0, 1]
    for i, item in enumerate(path_set):
        word_group = item.split('\\')
        if word_group[1] != tuple_tmp[0]:
            tuple_tmp[1] = 1
            tuple_tmp[0] = word_group[1]
        origin_path.append("data" + "\\" + word_group[1] + "\\" + word_group[2])
        target_path.append("valid" + "\\" + word_group[1])
        copyfile(origin_path[i], target_path[i])
        # 重命名为  类别_序号  的风格
        os.rename(target_path[i]+"\\"+word_group[2],target_path[i]+"\\"+word_group[1]+"_"+str(tuple_tmp[1])+".jpg")
        tuple_tmp[1] = tuple_tmp[1]+1

# data_to_valid()
# data_to_train()

