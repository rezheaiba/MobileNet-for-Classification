"""
# @Time    : 2023/2/8 13:43
# @File    : dataloader.py
# @Author  : rezheaiba
"""
import csv
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class Loader(Dataset):
    def __init__(self, csv_path, root, transforms_=None, mode="train"):
        self.name = []
        self.label1 = []
        self.label2 = []
        self.label3 = []
        self.label4 = []
        with open(csv_path, encoding='utf-8-sig') as f:
            for row in csv.reader(f, skipinitialspace=True):
                self.name.append(row[0])
                self.label1.append(int(row[1]))
                self.label2.append(int(row[2]))
                self.label3.append(int(row[3]))
                self.label4.append(int(row[4]))
        self.root = root
        self.mode = mode
        if transforms_ is not None:
            self.transform = transforms_

    def __getitem__(self, index):
        dir = self.name[index].split('_')[0]
        path = os.path.join(self.root, self.mode, dir, self.name[index])
        img = Image.open(path)
        label1 = self.label1[index]
        label2 = self.label2[index]
        label3 = self.label3[index]
        label4 = self.label4[index]

        if self.transform is not None:
            img = self.transform(img)
        return {"image": img,
                "label1": torch.tensor(label1, dtype=torch.int64),
                "label2": torch.tensor(label2, dtype=torch.int64),
                "label3": torch.tensor(label3, dtype=torch.int64),
                "label4": torch.tensor(label4, dtype=torch.int64)}

    def __len__(self):
        return len(self.name)


'''if __name__ == '__main__':
    dataloader = Loader(csv_path='./label/test_label.csv',
                        root='D:\Dataset\data\MTest',
                        transforms_=[transforms.Resize((224, 224)), transforms.ToTensor()],
                        mode='test')
    print(len(dataloader))
    data = DataLoader(dataloader,
                      batch_size=64,
                      shuffle=False,
                      num_workers=0)
    print(len(data))'''
