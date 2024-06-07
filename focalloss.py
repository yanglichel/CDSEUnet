

import torch
import torch.nn as nn
import numpy as np

# --------------------------- BINARY LOSSES ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #inputs = nn.Sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, inputs, targets,weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.inputs=inputs
        self.targets=targets

    def forward(self, smooth=1): #inputs, targets,
        inputs = nn.Sigmoid(self.inputs)
        inputs = inputs.view(-1)
        targets = self.targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE =nn.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class FocalLoss2(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss2, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        inputs = nn.Sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = nn.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


if __name__=='__main__':
    pred=torch.rand(16,1,28,28)
    y = torch.rand(16,1,28,28)
    #loss1=FocalLoss(pred,y)
    loss2 =FocalLoss(pred,y)

    #loss2=DiceLoss(pred,y)


    #print('loss1',loss1)
    print('loss2', loss2)

