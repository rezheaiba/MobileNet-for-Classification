"""
# @Time    : 2022/8/6 12:31
# @File    : obj-dec-vedio.py
# @Author  : rezheaiba
"""
from PIL import ImageFont, ImageDraw
# 导入中文字体，指定字号
font = ImageFont.truetype('SimHei.ttf', 32)

import os
import time
import shutil

import cv2
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签

import torch
import torch.nn.functional as F
from torchvision import models

import mmcv


# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# 后端绘图，不显示，只保存
import matplotlib
matplotlib.use('Agg')

from model_v2 import MobileNetV2
model = MobileNetV2(7)
model.load_state_dict(torch.load('../weight-final/MobileNetV2_final_best.pth', map_location=device))
model = model.eval()
model = model.to(device)


df = pd.read_csv('final_class_index.csv')
idx_to_labels = {}
for idx, row in df.iterrows():
    idx_to_labels[row['ID']] = [row['class'], row['Chinese']]

from torchvision import transforms

# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])


def pred_single_frame(img, n=5):
    '''
    输入摄像头画面bgr-array，输出前n个图像分类预测结果的图像bgr-array
    '''
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
    img_pil = Image.fromarray(img_rgb)  # array 转 pil
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

    top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析出置信度

    # 在图像上写字
    draw = ImageDraw.Draw(img_pil)
    # 在图像上写字
    for i in range(len(confs)):
        pred_class = idx_to_labels[pred_ids[i]][1]
        text = '{:<15} {:>.3f}'.format(pred_class, confs[i])
        # 文字坐标(左上为0,0->x,y)，中文字符串，字体，rgba颜色
        draw.text((50, 200 + 50 * i), text, font=font, fill=(255, 0, 0, 1))

    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # RGB转BGR

    return img_bgr, pred_softmax

input_video = r'D:\Dataset\data\飞絮\实景飞絮\10.20.40.8_008M_20221115174653000.mp4'

# 创建临时文件夹，存放每帧结果
temp_out_dir = time.strftime('%Y%m%d%H%M%S')
os.mkdir(temp_out_dir)
print('创建文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))

imgs = mmcv.VideoReader(input_video)

prog_bar = mmcv.ProgressBar(len(imgs))

# 对视频逐帧处理
for frame_id, img in enumerate(imgs):
    ## 处理单帧画面
    img, pred_softmax = pred_single_frame(img, n=5)





    # 将处理后的该帧画面图像文件，保存至 /tmp 目录下
    cv2.imwrite(f'{temp_out_dir}/{frame_id:06d}.jpg', img)

    prog_bar.update()  # 更新进度条

# 把每一帧串成视频文件
mmcv.frames2video(temp_out_dir, r'D:\Dataset\data\飞絮\实景飞絮\output\10.20.40.8_008M_20221115174653000.mp4', fps=imgs.fps, fourcc='mp4v')

shutil.rmtree(temp_out_dir)  # 删除存放每帧画面的临时文件夹
# os.removedirs(temp_out_dir)
print('删除临时文件夹', temp_out_dir)