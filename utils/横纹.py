"""
# @Time     : 2022/8/23 10:53
# @File     : 横纹3.py
# @Author   : rezheaiba
"""
import math
import os

import cv2
import numpy as np


def hengwen(img):
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    period = np.random.randint(6, 9)
    if period % 2 == 0:
        pass
    else:
        period += 1
    start_point = np.random.randint(0, h // 8)
    ratio = np.random.uniform(0.1, 0.3)

    k = period * math.pi / h
    value = np.arange(h - start_point)
    up_left = np.arange(h - start_point, h)
    y = [k * x for x in value]
    y_left = [k * x for x in up_left]
    base = 1
    # sin = [1 + ratio *  math.sin(yy) for yy in y]
    sin = []
    for i in range(h):
        if i < start_point:
            sin.append(base + ratio * math.sin(y_left[i]))
        if i >= start_point:
            sin.append(base + ratio * math.sin(y[i - start_point]))
    sin = [round(x, 3) for x in sin]
    # print(sin)
    for i in range(h):
        a = np.zeros((1, w, c))
        a = img[i:i + 1, :, :] * sin[i]
        a[a > 255] = 255
        img[i:i + 1, :, :] = a

    return img


if __name__ == '__main__':
    image_path = r'C:\Users\A\Desktop\a'
    image_out_path = r'C:\Users\A\Desktop\a1'
    if not os.path.exists(image_out_path):
        os.makedirs(image_out_path, exist_ok=True)
    list = os.listdir(image_path)
    for i in range(len(list)):
        img = cv2.imread(os.path.join(image_path, list[i]))
        result = hengwen(img)
        cv2.imwrite(os.path.join(image_out_path, list[i]), result)
