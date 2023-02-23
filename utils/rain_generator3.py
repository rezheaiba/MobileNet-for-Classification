"""
# @Time    : 2022/8/12 14:33
# @File    : rain_generator.py
# @Author  : rezheaiba
"""
"""
# @Time     : 2022/8/11 11:16
# @File     : rain.py
# @Author   : rezheaiba
"""
import os

from cv2 import cv2
import numpy as np


# 首先，我们需要生成不同密度的随机噪声来模拟不同大小的雨量，于是利用了下面的函数来生成。
# 主要的使用了均匀随机数和阈值来控制噪声的水平。由于生成噪声是浮点数，所以在value上乘了尺度缩小因子。
def get_noise(img, value=10):
    '''
    #生成噪声图像
    >> 输入： img图像

        value= 大小控制雨滴的多少
    >> 返回图像大小的模糊噪声图像
    '''

    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value * 0.01
    # 置黑
    noise[np.where(noise < (256 - v))] = 0

    noise[noise > 253] = 255

    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 16, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    # 可以输出噪声看看
    '''cv2.imshow('img',noise)
    cv2.waitKey()
    cv2.destroyWindow('img')'''
    return noise


# 随后，需要对噪声拉长、旋转方向，模拟不同大小和方向的雨水。
def rain_blur(noise, noise2, length=10, length2=10, angle=0, w=1):
    '''
    将噪声加上运动模糊,模仿雨滴

    >>输入
    noise：输入噪声图，shape = img.shape[0:2]
    length: 对角矩阵大小，表示雨滴的长度
    angle： 倾斜的角度，逆时针为正
    w:      雨滴大小

    >>输出带模糊的噪声

    '''

    # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # 生成对角矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
    k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

    # k = k / length                         #是否归一化

    blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

    # 转换到0-255区间
    # cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)

    # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans2 = cv2.getRotationMatrix2D((length2 / 2, length2 / 2), angle - 45, 1 - length / 100.0)
    dig2 = np.diag(np.ones(length2))  # 生成对角矩阵
    k2 = cv2.warpAffine(dig2, trans2, (length2, length2))  # 生成模糊核
    k2 = cv2.GaussianBlur(k2, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

    # k = k / length                         #是否归一化

    blurred2 = cv2.filter2D(noise2, -1, k2)  # 用刚刚得到的旋转后的核，进行滤波

    blurred = blurred2 + blurred
    # 转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)

    blurred = np.array(blurred, dtype=np.uint8)
    '''
    cv2.imshow('img',blurred)
    cv2.waitKey()
    cv2.destroyWindow('img')'''

    return blurred


def alpha_rain(rain, img, alpha=0.6, beta=0.8):
    # 输入雨滴噪声和图像
    # beta = 0.8   #results weight
    # 显示下雨效果

    # expand dimensin
    # 将二维雨噪声扩张为三维单通道
    # 并与图像合成在一起形成带有alpha通道的4通道图像
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel

    rain_result = img.copy()  # 拷贝一个掩膜
    rain_result = alpha * rain_result
    rain = np.array(rain, dtype=np.float32)  # 数据类型变为浮点数，后面要叠加，防止数组越界要用32位
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    # 对每个通道先保留雨滴噪声图对应的黑色（透明）部分，再叠加白色的雨滴噪声部分（有比例因子）
    return rain_result
    # cv2.imshow('rain_effct_result', rain_result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


def add_rain(rain, img, alpha, beta):
    # 输入雨滴噪声和图像
    # alpha：原图比例因子
    # 显示下雨效果

    # chage rain into  3-dimenis
    # 将二维rain噪声扩张为与原图相同的三通道图像
    rain = np.expand_dims(rain, 2)
    rain = np.repeat(rain, 3, 2)

    # 加权合成新图
    result = cv2.addWeighted(img, alpha, rain, beta, 1)
    return result
    # cv2.imshow('rain_effct', result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = r'D:\Dataset\data\genrain\rainy_5'
    image_out_path = r'D:\Dataset\data\genrain\rainy_5'  # rainy4修改了雨丝的亮度和宽度
    if not os.path.exists(image_out_path):
        os.makedirs(image_out_path, exist_ok=True)
    list = os.listdir(image_path)
    for l in list:
        noise_random = np.random.randint(180, 250)
        length_random = np.random.randint(150, 190)
        angle_random = np.random.randint(-30, 40)
        alpha_random = np.random.uniform(0.5, 0.9)
        beta_random = np.random.uniform(0.7, 0.8)
        print(os.path.join(image_path, l))

        img = cv2.imread(os.path.join(image_path, l))
        # 随机噪声
        noise = get_noise(img, value=1)
        noise2 = get_noise(img, value=1)
        # 噪声调整
        rain = rain_blur(noise, noise2, length=length_random, length2=90, angle=angle_random, w=11)  # angle取[-25,25]
        # 图片融合
        rain1 = alpha_rain(rain, img, alpha=0.9, beta=beta_random)  # 方法一，透明度赋值


        # cv2.imwrite(os.path.join(image_out_path, os.path.splitext(l)[0] + '-1.jpg'), rain1)

        angle_random = np.random.randint(-30, 40)
        length_random = np.random.randint(140, 200)
        alpha_random = np.random.uniform(0.8, 0.9)
        beta_random = np.random.uniform(0.65, 0.8)
        # 随机噪声
        noise = get_noise(img, value=1)
        noise2 = get_noise(img, value=1)
        # 噪声调整
        rain = rain_blur(noise, noise2, length=length_random, length2=90, angle=angle_random, w=11)  # angle取[-25,25]
        rain2 = add_rain(rain, img, alpha=alpha_random, beta=beta_random)

        noise3 = get_noise(img, value=1)
        noise4 = get_noise(img, value=0)
        # 噪声调整
        rain_ = rain_blur(noise3, noise4, length=length_random, length2=90, angle=angle_random, w=9)  # angle取[-25,25]
        rain2 = add_rain(rain_, rain2, alpha=1, beta=1.3)

        cv2.imwrite(os.path.join(image_out_path, os.path.splitext(l)[0] + '-2.jpg'), rain2)
        # break
