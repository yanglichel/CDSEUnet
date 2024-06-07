import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])




class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path1 = os.path.join(self.path, 'Images1', segment_name)
        # image_path2 = os.path.join(self.path, 'Images2', segment_name)
        # image_path2 = os.path.join(self.path, 'Images2_sobel1', segment_name)
        # image_path2 = os.path.join(self.path, 'Images2_prewitt1', segment_name)
        # image_path2 = os.path.join(self.path, 'canny99', segment_name)
        # image_path3 = os.path.join(self.path, 'canny77', segment_name)
        image_path4 = os.path.join(self.path, 'canny55', segment_name)
        # image_path5 = os.path.join(self.path, 'canny33', segment_name)
        # image_path2 = os.path.join(self.path, 'Images2_roberts1', segment_name)

        segment_image = keep_image_size_open_rgb(segment_path)
        image1 = keep_image_size_open_rgb(image_path1)
        # image2 = keep_image_size_open_rgb(image_path2)
        # image3 = keep_image_size_open_rgb(image_path3)
        image4 = keep_image_size_open_rgb(image_path4)
        # image5 = keep_image_size_open_rgb(image_path5)
        # print("*****",image1.size,image2.size)
        # image = torch.cat(
        #     (transform(image1), transform(image2), transform(image3), transform(image4), transform(image5)), dim=0)
        image = torch.cat(
            (transform(image1), transform(image4)), dim=0)

        # print(":::::::",image.size)
        # return transform(image), torch.Tensor(np.array(segment_image))
        return image, transform(segment_image)


if __name__ == '__main__':
    #from torch.nn.functional import one_hot
    #data = MyDataset('data')
    data = MyDataset(r'E:\pw_2024\PycharmProjects\data\coronacases\traindata')
   #data = MyDataset2(r'D:\PycharmProjects\dcvoid19-UNet\coronacases\valdata')

    print(data[0][0].shape)
    print(data[0][1].shape)
    #out=one_hot(data[0][1].long())
    #print(out.shape)
