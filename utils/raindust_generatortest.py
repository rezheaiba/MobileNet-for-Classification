"""
# @Time    : 2022/8/12 14:33
# @File    : dust_generator.py
# @Author  : rezheaiba
"""
import math

"""
# @Time     : 2022/8/11 11:16
# @File     : dust.py
# @Author   : rezheaiba
"""
import os

from cv2 import cv2
import numpy as np
from imagecorruptions import corrupt


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
    # cv2.imshow('img',noise)
    # cv2.waitKey()
    # cv2.destroyWindow('img')
    return noise


# 随后，需要对噪声拉长、旋转方向，模拟不同大小和方向的雨水。
def dust_blur(noise, noise2, noise3, length=10, length2=10, length3=10, angle1=0, angle2=0, angle3=0, w=1, w2=1, w3=1):
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
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle1 - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # 生成对角矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
    k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

    # k = k / length                         #是否归一化

    blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

    # 转换到0-255区间
    # cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)

    # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans2 = cv2.getRotationMatrix2D((length2 / 2, length2 / 2), angle2 - 45, 1 - length2 / 100.0)
    dig2 = np.diag(np.ones(length2))  # 生成对角矩阵
    k2 = cv2.warpAffine(dig2, trans2, (length2, length2))  # 生成模糊核
    k2 = cv2.GaussianBlur(k2, (w2, w2), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

    # k = k / length                         #是否归一化

    blurred2 = cv2.filter2D(noise2, -1, k2)  # 用刚刚得到的旋转后的核，进行滤波

    trans3 = cv2.getRotationMatrix2D((length3 / 2, length3 / 2), angle3 - 45, 1 - length3 / 100.0)
    dig3 = np.diag(np.ones(length3))  # 生成对角矩阵
    k3 = cv2.warpAffine(dig3, trans3, (length3, length3))  # 生成模糊核
    k3 = cv2.GaussianBlur(k3, (w3, w3), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

    # k = k / length                         #是否归一化

    blurred3 = cv2.filter2D(noise3, -1, k3)  # 用刚刚得到的旋转后的核，进行滤波

    blurred = blurred3 + blurred2 + blurred
    # 转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)

    blurred = np.array(blurred, dtype=np.uint8)
    '''
    cv2.imshow('img',blurred)
    cv2.waitKey()
    cv2.destroyWindow('img')'''

    return blurred


def alpha_dust(dust, img, alpha=0.6, beta=0.8):
    # 输入雨滴噪声和图像
    # beta = 0.8   #results weight
    # 显示下雨效果

    # expand dimensin
    # 将二维雨噪声扩张为三维单通道
    # 并与图像合成在一起形成带有alpha通道的4通道图像
    dust = np.expand_dims(dust, 2)
    dust_effect = np.concatenate((img, dust), axis=2)  # add alpha channel

    dust_result = img.copy()  # 拷贝一个掩膜
    dust_result = alpha * dust_result
    dust = np.array(dust, dtype=np.float32)  # 数据类型变为浮点数，后面要叠加，防止数组越界要用32位

    dust_result[:, :, 0] = dust_result[:, :, 0] * (255 - dust[:, :, 0]) / 255.0 + beta * dust[:, :, 0]
    dust_result[:, :, 1] = dust_result[:, :, 1] * (255 - dust[:, :, 0]) / 255 + beta * dust[:, :, 0]
    dust_result[:, :, 2] = dust_result[:, :, 2] * (255 - dust[:, :, 0]) / 255 + beta * dust[:, :, 0]
    # 对每个通道先保留雨滴噪声图对应的黑色（透明）部分，再叠加白色的雨滴噪声部分（有比例因子）
    return dust_result
    # cv2.imshow('dust_effct_result', dust_result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


def transform_dust(img, ratio):  # 鱼眼效果
    rows, cols, c = img.shape
    center_x, center_y = rows / ratio, cols / ratio
    print('ratio=', ratio)
    # radius = min(center_x,center_y)
    radius = math.sqrt(rows ** 2 + cols ** 2) / 2
    new_img = img.copy()
    for i in range(rows):
        for j in range(cols):
            dis = math.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if dis <= radius:
                new_i = np.int(np.round(dis / radius * (i - center_x) + center_x))
                new_j = np.int(np.round(dis / radius * (j - center_y) + center_y))
                new_img[i, j] = img[new_i, new_j]
                # print((i,j),'\t',(new_i,new_j))
    return new_img


def add_dust(dust, img, alpha, beta):
    # 输入雨滴噪声和图像
    # alpha：原图比例因子
    # 显示下雨效果

    # chage dust into  3-dimenis
    # 将二维dust噪声扩张为与原图相同的三通道图像
    dust = np.expand_dims(dust, 2)
    dust = np.repeat(dust, 3, 2)
    dust = corrupt(dust, corruption_name='defocus_blur', severity=4)

    dust = transform_dust(dust, ratio=np.random.uniform(1,8))
    # 加权合成新图
    result = cv2.addWeighted(img, alpha, dust, beta, 1)
    return result


if __name__ == '__main__':
    image_path = r'../img1'
    image_out_path = r'../img1'
    if not os.path.exists(image_out_path):
        os.makedirs(image_out_path, exist_ok=True)
    list = os.listdir(image_path)
    idx = 0
    for l in list:
        angle_random1 = np.random.randint(-90, 90)
        angle_random2 = np.random.randint(-90, 90)
        angle_random3 = np.random.randint(-90, 90)

        length_random = np.random.randint(140, 170)
        alpha_random = np.random.uniform(0.8, 0.9)
        beta_random = np.random.uniform(0.9, 2.5)

        print(os.path.join(image_path, l))
        img = cv2.imread(os.path.join(image_path, l))
        # 随机噪声
        noise_random = np.random.uniform(0.3, 0.35)
        noise = get_noise(img, value=noise_random)
        noise2 = get_noise(img, value=noise_random)
        noise3 = get_noise(img, value=noise_random)

        # 噪声调整
        dust = dust_blur(noise, noise2, noise3, length=length_random, length2=120, length3=120, angle1=angle_random1,
                         angle2=angle_random2, angle3=angle_random3, w=5, w2=13, w3=7)  # angle取[-25,25]
        # 图片融合
        dust1 = add_dust(dust, img, alpha=alpha_random, beta=beta_random)

        cv2.imwrite(os.path.join(image_out_path, os.path.splitext(l)[0] + '-1.jpg'), dust1)

        angle_random1 = np.random.randint(-90, 90)
        angle_random2 = np.random.randint(-90, 90)
        angle_random3 = np.random.randint(-90, 90)

        length_random = np.random.randint(140, 170)
        alpha_random = np.random.uniform(0.8, 0.9)
        # beta_random = np.random.uniform(0.85, 0.95)
        beta_random = np.random.uniform(0.9, 2.5)

        # 随机噪声
        noise_random = np.random.uniform(0.3, 0.35)
        noise = get_noise(img, value=noise_random)
        noise2 = get_noise(img, value=noise_random)
        noise3 = get_noise(img, value=noise_random)
        # 噪声调整
        dust = dust_blur(noise, noise2, noise3, length=length_random, length2=120, length3=120, angle1=angle_random1,
                         angle2=angle_random2, angle3=angle_random3, w=7, w2=13, w3=7)
        dust2 = add_dust(dust, img, alpha=alpha_random, beta=beta_random)  # 方法二,加权后有玻璃外的效果,alpha调低后，噪声的黑色会显现，可以模拟黑夜
        cv2.imwrite(os.path.join(image_out_path, os.path.splitext(l)[0] + '-2.jpg'), dust2)
        # if idx == 10:
        #     break
        # idx +=1
