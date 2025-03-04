import os
import shutil
import warnings

if __name__ == '__main__':
    # 1. 对train检验文件是否存在
    with open("list\\train.txt") as f:
        train_list = f.read().splitlines()
    # 2. 移动A, B, labels文件到train里面
    for file in train_list:
        path_A = os.path.join("A", file)
        path_B = os.path.join("B", file)
        path_label = os.path.join("labels", file)
        A_exist = os.path.exists(path_A)
        B_exist = os.path.exists(path_B)
        Label_exist = os.path.exists(path_label)
        if A_exist and B_exist and Label_exist:
            shutil.move(path_A, os.path.join("train", path_A))
            shutil.move(path_B, os.path.join("train", path_B))
            shutil.move(path_label, os.path.join("train", path_label))
        else:
            warnings.warn(f"Image not exist in train: A: {A_exist}, B: {B_exist}, label: {Label_exist}")

    # 1. 对val检验文件是否存在
    with open("list\\val.txt") as f:
        val_list = f.read().splitlines()
    # 2. 移动A, B, labels文件到val里面
    for file in val_list:
        path_A = os.path.join("A", file)
        path_B = os.path.join("B", file)
        path_label = os.path.join("labels", file)
        A_exist = os.path.exists(path_A)
        B_exist = os.path.exists(path_B)
        Label_exist = os.path.exists(path_label)
        if A_exist and B_exist and Label_exist:
            shutil.move(path_A, os.path.join("val", path_A))
            shutil.move(path_B, os.path.join("val", path_B))
            shutil.move(path_label, os.path.join("val", path_label))
        else:
            warnings.warn(f"Image not exist in val: A: {A_exist}, B: {B_exist}, label: {Label_exist}")
