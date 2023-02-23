"""
# @Time    : 2022/8/3 14:03
# @File    : dataAugmentation.py
# @Author  : rezheaiba
"""
import os
import shutil
import random
from cv2 import cv2
import numpy as np

'''opencv数据增强
    对图片进行色彩增强、高斯噪声、水平镜像、放大、旋转、剪切
'''


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


def gasuss_noise(image, path_out_gasuss, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    cv2.imwrite(path_out_gasuss, out)


def mirror(image, path_out_mirror):
    '''
        水平镜像
    '''
    h_flip = cv2.flip(image, 1)
    cv2.imwrite(path_out_mirror, h_flip)


def resize(image, path_out_large):
    '''
        放大两倍
    '''
    height, width = image.shape[:2]
    large = cv2.resize(image, (352, 288))
    cv2.imwrite(path_out_large, large)


def rotate(image, path_out_rotate):
    '''
        旋转
    '''
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    dst = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(path_out_rotate, dst)


def shear(image, path_out_shear):
    '''
        剪切
    '''
    height, width = image.shape[:2]
    cropped = image[int(height / 9):height, int(width / 9):width]
    cv2.imwrite(path_out_shear, cropped)


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


def randomcrop(img, path_out_crop, scale=0.8):
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
    cv2.imwrite(path_out_crop, resized)


def main():
    image_path = r'C:\Users\A\Desktop\b'
    image_out_path = r'C:\Users\A\Desktop\c'
    if not os.path.exists(image_out_path):
        os.makedirs(image_out_path, exist_ok=True)
    list = os.listdir(image_path)
    print(list)
    print("----------------------------------------")
    print("The original data path:" + image_path)
    print("The original data set size:" + str(len(list)))
    print("----------------------------------------")

    imageNameList = [
        '_color.jpg',
        '_gasuss.jpg',
        '_mirror.jpg',
        '_large.jpg',
        '_rotate.jpg',
        '_shear.jpg',
        '_colorb.jpg',
        '_colorc.jpg',
        '_crop.jpg',
        '.jpg']
    for i in range(0, len(list)):
        path = os.path.join(image_path, list[i])
        out_image_name = os.path.splitext(list[i])[0]
        for j in range(0, len(imageNameList)):
            path_out = os.path.join(
                image_out_path, out_image_name + imageNameList[j])
            image = cv2.imread(path)
            if j == 0:
                # contrast_brightness_image(image, 1.2, 10, path_out)
                continue
            elif j == 1:
                # gasuss_noise(image, path_out)  # 模糊不太需要
                continue
            elif j == 2:
                # mirror(image, path_out)
                continue
            elif j == 3:
                # resize(image, path_out)  # 默认使用双线性插值法，送入不同resolution的图片，增强网络的学习能力
                continue
            elif j == 4:
                # rotate(image, path_out)
                continue
            elif j == 5:
                # shear(image, path_out)  # 用randomcrop取代
                continue
            elif j == 6:
                # colorjitter(image, path_out, cj_type='b')  # 有的图可以用，hdr不能用
                continue
            elif j == 7:
                # colorjitter(image, path_out, cj_type='c')
                continue
            elif j == 8:
                randomcrop(image, path_out, )
                # continue
            else:
                # shutil.copy(path, path_out)
                continue
        print(out_image_name + "success！", end='\t')
    print("----------------------------------------")
    print("The data augmention path:" + image_out_path)
    outlist = os.listdir(image_out_path)
    print("The data augmention sizes:" + str(len(outlist)))
    print("----------------------------------------")
    print("Rich sample for:" + str(len(outlist) - len(list)))


if __name__ == '__main__':
    main()
