"""
# @Time    : 2022/11/11 17:13
# @File    : imgaug雨.py
# @Author  : rezheaiba
"""
# ##############简易变换#################

# https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html
from cv2 import cv2
from imgaug import augmenters as iaa
import os

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
# sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# 定义一组变换方法.
seq = iaa.Sequential([
    # iaa.MotionBlur(k=15),  # 运动模糊
    # iaa.Clouds(),  # 云雾
    # iaa.imgcorruptlike.Fog(severity=1),  # 多雾/霜
    # iaa.imgcorruptlike.Snow(severity=1),  # 下雨、大雪
    iaa.Rain(drop_size=(0.10, 0.20), speed=(0.2, 0.3)),  # 雨
    # iaa.Rain(drop_size=(0.8, 0.8), speed=(0.25, 0.35)),  # 雨
    # iaa.MotionBlur(k=15),  # 运动模糊

    # iaa.Snowflakes(flake_size=(0.9, 0.9), speed=(0.01, 0.02)), # 雪点
    # iaa.MotionBlur(k=10)
    # iaa.imgcorruptlike.Spatter(severity=2),  # 溅 123水滴、45泥
    # iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),# 对比度变为原来的一半或者二倍
    # iaa.imgcorruptlike.Brightness(severity=2),  # 亮度增加
    # iaa.imgcorruptlike.Saturate(severity=3),  # 色彩饱和度
    # iaa.FastSnowyLandscape(lightness_threshold=(100, 255),lightness_multiplier=(1.5, 2.0)), # 雪地   亮度阈值是从 uniform(100, 255)（每张图像）和来自 uniform(1.5, 2.0)（每张图像）的乘数采样的。
    # iaa.Cartoon(blur_ksize=3, segmentation_size=1.0, saturation=2.0, edge_prevalence=1.0), # 卡通

])
path = r'2.jpg'
savedpath = r'3.jpg'
img = cv2.imread(path)
images_aug = seq.augment_image(img)
cv2.imwrite(os.path.join(savedpath), images_aug)

# 图片文件相关路径
path = r'D:\Dataset\data\genrain\c'
savedpath = r'D:\Dataset\data\genrain\rainy_5'
if not os.path.exists(savedpath):
    os.makedirs(savedpath, exist_ok=True)

for file in os.listdir(path):
    img = cv2.imread(os.path.join(path, file))
    _img=seq.augment_image(img)
    cv2.imwrite(os.path.join(savedpath, file), _img)





# imglist = []
# filelist = os.listdir(path)
#
# for item in filelist:
#     img = cv2.imread(os.path.join(path, item))
#     imglist.append(img)
#
# images_aug = seq.augment_images(imglist)
# for index in range(len(images_aug)):
#     filename = str(filelist[index])
#     cv2.imwrite(os.path.join(savedpath , filename), images_aug[index])