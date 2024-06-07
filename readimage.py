import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from PIL import Image
import numpy as np
import torch

# dog.jpg    width = 1599, height=1066, channel=3

'''
# 使用skimage读取图像
img_skimage = io.imread('dog.jpg')        # skimage.io imread()-----np.ndarray,  (H x W x C), [0, 255],RGB
print(img_skimage.shape)

# 使用opencv读取图像
img_cv = cv2.imread('dog.jpg')            # cv2.imread()------np.array, (H x W xC), [0, 255], BGR
print(img_cv.shape)

'''
# 使用PIL读取
img_pil = Image.open('n290.png')         # PIL.Image.Image对象
img_pil_1 = np.array(img_pil)           # (H x W x C), [0, 255], RGB
print(img_pil_1)

plt.figure()
#for i, im in enumerate([img_skimage, img_cv, img_pil_1]):
for i, im in enumerate([ img_pil_1]):
    ax = plt.subplot(1, 3, i + 1)
    ax.imshow(im)
    plt.pause(0.01)


