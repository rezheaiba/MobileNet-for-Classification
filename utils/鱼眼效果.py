"""
# @Time    : 2022/12/30 15:25
# @File    : 鱼眼效果.py
# @Author  : rezheaiba
"""
from cv2 import cv2
import numpy as np
import math
def transform(img):
    ratio = np.random.uniform(0,10)
    ratio = 1
    print(ratio)
    rows,cols,c = img.shape
    center_x,center_y = rows/ratio,cols/ratio
    #radius = min(center_x,center_y)
    radius = math.sqrt(rows**2+cols**2)/2
    new_img = img.copy()
    for i in range(rows):
        for j in range(cols):
            dis = math.sqrt((i-center_x)**2+(j-center_y)**2)
            if dis <= radius:
                new_i = np.int(np.round(dis/radius*(i-center_x)+center_x))
                new_j = np.int(np.round(dis/radius*(j-center_y)+center_y))
                new_img[i,j] = img[new_i,new_j]
                #print((i,j),'\t',(new_i,new_j))
    return new_img
img = cv2.imread('src.jpg')
img = transform(img)
cv2.imwrite('src.jpg', img)
