import glob
import os

from cv2 import cv2
from skimage.metrics import structural_similarity

real_path = r'D:\Dataset\data\final\train\outdoor'
# real_path = r'D:\Dataset\data\genrain\rainy_1'
# real_path = r'../../final/train/outdoor'

if __name__ == '__main__':
    real_name_lists = glob.glob(os.path.join(real_path, '*.jpg'))
    for idx, img_1 in enumerate(real_name_lists):  # 非顺序，不过都是pair
        real_img = cv2.imread(img_1)
        if real_img.shape[1] > 1000 or real_img.shape[0] > 1000:
            print('img_1:{}'.format(str(img_1)))
