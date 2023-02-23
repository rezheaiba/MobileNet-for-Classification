"""
# @Time    : 2022/8/8 11:11
# @File    : test.py
# @Author  : rezheaiba
"""
import random

import numpy as np
from cv2 import cv2


def randomcrop(img, scale=0.6):
    '''
    ### Random Crop ###
    img: image
    gt_boxes: format [[obj x1 y1 x2 y2],...]
    scale: percentage of cropped area
    '''

    # Crop image
    height, width = int(img.shape[0] * scale), int(img.shape[1] * scale)
    x = random.randint(0, img.shape[1] - int(width))
    y = random.randint(0, img.shape[0] - int(height))
    cropped = img[y:y + height, x:x + width]
    resized = cv2.resize(cropped, (img.shape[1], img.shape[0]))

    return resized


def colorjitter(img, path_out_color, cj_type="b", ):  # 亮度、对比度和饱和度
    '''
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: constast}
    '''
    if cj_type == "b":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(path_out_color, img)

    elif cj_type == "c":
        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(img)
        dummy = dummy * (contrast / 127 + 1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        cv2.imwrite(path_out_color, img)


def contrast_brightness_image(src1, a, g, path_out):
    '''
        色彩增强（通过调节对比度和亮度）
    '''
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    # addWeighted函数说明:计算两个图像阵列的加权和
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)
    cv2.imwrite(path_out, dst)


if __name__ == '__main__':
    img = cv2.imread('../images/img.jpg')
    colorjitter(img, '../images/8-1.jpg', 'b')
    colorjitter(img, '../images/8-2.jpg', 'c')
    contrast_brightness_image(img, 1.2, 10, '../images/8-3.jpg')
