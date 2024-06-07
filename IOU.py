import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import matplotlib.pyplot as plt

# prediction
predict = torch.tensor(
    [0.01, 0.03, 0.02, 0.02, 0.05, 0.12, 0.09, 0.07, 0.89, 0.85, 0.88, 0.91, 0.99, 0.97, 0.95, 0.97]).reshape(1, 1, 4,
                                                                                                              4)
'''
tensor([[[[0.0100, 0.0300, 0.0200, 0.0200],
          [0.0500, 0.1200, 0.0900, 0.0700],
          [0.8900, 0.8500, 0.8800, 0.9100],
          [0.9900, 0.9700, 0.9500, 0.9700]]]])
'''

# label
label = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, 1, 4, 4)
'''
tensor([[[[0, 0, 0, 0],
          [0, 0, 0, 0],
          [1, 1, 1, 1],
          [1, 1, 1, 1]]]])
'''


# Dice
def Dice(pred, true):
    intersection = pred * true  # 计算交集  pred ∩ true
    temp = pred + true  # pred + true
    smooth = 1e-8  # 防止分母为 0
    dice_score = 2 * intersection.sum() / (temp.sum() + smooth)
    return dice_score


print(f"dice:{Dice(predict,label)}")

# Iou
def Iou(pred, true):
    intersection = pred * true  # 计算交集  pred ∩ true
    temp = pred + true  # pred + true
    union = temp - intersection  # 计算并集：A ∪ B = A + B - A ∩ B
    smooth = 1e-8  # 防止分母为 0
    iou_score = intersection.sum() / (union.sum() + smooth)
    return iou_score

print(f"Iou:{Iou(predict,label)}")

# dice 和 iou 的换算
def dice_and_iou(x):
    y = x / (2 - x)
    return y

print(f"dice_and_iou:{dice_and_iou(Dice(predict,label))}")
dice = np.arange(0, 1, 0.001)
print(dice.shape)
iou = dice_and_iou(dice)
print(iou.shape)
plt.plot(dice, iou)
plt.xlabel('dice')
plt.ylabel('iou')
plt.show()
