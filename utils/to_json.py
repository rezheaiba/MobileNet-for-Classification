"""
# @Time    : 2022/8/1 12:16
# @File    : to_json.py
# @Author  : rezheaiba
"""
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

data_root = r'D:\Dataset'  # get data root path
image_path = os.path.join(data_root, "places365_standard")  # flower data set path
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
train_num = len(train_dataset)  # 67

flower_list = train_dataset.class_to_idx  # 类别和index的映射，这是ImageFolder处理后的一个参数
cla_dict = dict((val, key) for key, val in flower_list.items())  # 映射存为字典格式
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)  # 把数据转json文件，indent：数据格式缩进显示4个空格，读起来更加清晰
with open('place365_class_indices.json', 'w') as json_file:
    json_file.write(json_str)
