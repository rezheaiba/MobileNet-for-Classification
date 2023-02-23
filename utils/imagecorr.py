"""
# @Time    : 2022/8/9 9:52
# @File    : imagecorr.py
# @Author  : rezheaiba
"""
import numpy as np
from imagecorruptions import corrupt
from cv2 import cv2

image = cv2.imread('2.jpg', cv2.COLOR_BGR2RGB)
image = np.asarray(image)

corrupted_image = corrupt(image, corruption_name='fog', severity=3)

cv2.imshow('1', corrupted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()