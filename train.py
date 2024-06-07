import os
import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data5 import *
#from data2 import *
#from net import *
#from mynet2 import *

from CDSEUnet import *
#from cbam_net import *
from focalloss import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

weight_path = 'params/new_threshold.pth'
train_data_path =r'E:\pw_2024\PycharmProjects\data\coronacases\traindata'
val_data_path = r'E:\pw_2024\PycharmProjects\data\coronacases\valdata'
save_path = 'train_image'


def Dice(pred, true):
    intersection = pred * true  # 计算交集  pred ∩ true
    temp = pred + true  # pred + true
    smooth = 1e-8  # 防止分母为 0
    dice_score = 2 * intersection.sum() / (temp.sum() + smooth)
    return dice_score

def Iou(pred, true):
    intersection = pred * true  # 计算交集  pred ∩ true
    temp = pred + true  # pred + true
    union = temp - intersection  # 计算并集：A ∪ B = A + B - A ∩ B
    smooth = 1e-8  # 防止分母为 0
    iou_score = intersection.sum() / (union.sum() + smooth)
    return iou_score


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss_fun1 = nn.BCELoss()
        dice_loss1=(1 - Dice(y, x))
        myloss=loss_fun1(x,y)*0.5+dice_loss1*0.5
        return myloss

import cv2
def res(x):
    a=x.cpu()
    a = a.detach().numpy()
    k = np.zeros((a.shape[0], a.shape[1], a.shape[2], a.shape[3]))

    w = (a.shape[0])
    for i in range(w):
        b0 = a[i, 0]
        b = b0 * 255.0
        b = np.trunc(b)
        b = b.astype('uint8')
        ret, result1 = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        temp = b0[:, :] > ret / 255.0
        s = np.where(temp, 1, 0)
        s = s.astype('uint8')
        k[i, 0] = s

    k=torch.from_numpy(k)
    k=k.cuda()
    return k





if __name__ == '__main__':
    #num_classes = 2 + 1  # +1是背景也为一类
    train_data_loader = DataLoader(MyDataset(train_data_path), batch_size=4, shuffle=True)
    val_data_loader= DataLoader(MyDataset(val_data_path), batch_size=4, shuffle=True)
    #net = UNet(num_classes).to(device)
    net = UNet().to(device)

    #net=cbam_UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    lr = 0.001
    learning_rate_decay_start = 20
    learning_rate_decay_every = 10
    learning_rate_decay_rate = 0.9

    opt = optim.Adam(net.parameters(),lr=0.001)
    #opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-4)

    #opt = optim.Adam(net.parameters())
    #loss_fun = nn.CrossEntropyLoss()
    loss_fun = nn.BCELoss()


    max_dice = 0
    max_epoch = 0
    epoch = 1



    while epoch < 300:
        train_num_correct = 0
        train_num_pixels = 0
        val_num_correct = 0
        val_num_pixels = 0
        train_Dice = 0
        train_Iou = 0
        val_Dice = 0
        val_Iou = 0
        train_cnt = 0
        val_cnt = 0

        net.train()
        '''
        if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
            frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
            decay_factor = learning_rate_decay_rate ** frac
            current_lr = lr * decay_factor
            for group in opt.param_groups:
                group['lr'] = current_lr        else:
            current_lr = lr
        print('learning_rate: %s' % str(current_lr))
        '''

        for i, (image, segment_image) in enumerate(tqdm.tqdm(train_data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            #train_loss = loss_fun(out_image, segment_image.long())
            train_loss = loss_fun(out_image,segment_image)         # *0.2+(1-Dice(out_image, segment_image))*0.8

            opt.zero_grad()
            train_loss.backward()
            opt.step()



            #out_image = 1 if out_image>0.5 else 0
            #out_image=(out_image>0.5).float()
            out_image=res(out_image)
            segment_image=res(segment_image)
            '''

            x2 = torch.randn(4,1,256,256)
            x2[0] = res[0]
            x2[1] = res[1]
            x2[2] = res[2]
            x2[3] = res[3]
            out_image = (out_image > x2).float()
            '''

            train_num_correct +=(out_image== segment_image).sum()
            train_num_pixels += torch.numel(out_image)


            _image = image[0]
            #_segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            #_out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255
            _segment_image =segment_image[0]
            _out_image =out_image[0]
            img = torch.stack([_segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')


            out_image=out_image.reshape(1,-1)
            segment_image=segment_image.reshape(1,-1)



            train_Dice += Dice(out_image, segment_image)

            # train_Dice+=Dice(out_image, segment_image)
            #train_Iou += Iou(out_image, segment_image)

            train_cnt +=1

        print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
        #print(f'num_pixels===>>{train_num_correct, train_num_pixels}')
        #print(f'{epoch}-{i}-train_acc===>>{train_num_correct/train_num_pixels*100}')
        print(f'{epoch}-{i}-train_Dice===>>{train_Dice/train_cnt}')
        #print(f'{epoch}-{i}-train_IOU===>>{train_Iou/train_cnt}')
        #if epoch % 10 == 0:
        #    torch.save(net.state_dict(), weight_path)
        #    print('save successfully!')

        net.eval()
        with torch.no_grad():
            for i, (image, segment_image) in enumerate(tqdm.tqdm(val_data_loader)):
                image, segment_image = image.to(device), segment_image.to(device)
                out_image = net(image)
                # train_loss = loss_fun(out_image, segment_image.long())
                #val_loss = loss_fun(out_image, segment_image)
                # out_image = 1 if out_image>0.5 else 0

                #out_image = (out_image > (0.5).cuda()).float()
                out_image = res(out_image)
                segment_image = res(segment_image)

                '''

                x2 = torch.randn(4, 1, 256, 256)
                x2[0] = res[0]
                x2[1] = res[1]
                x2[2] = res[2]
                x2[3] = res[3]
                out_image = (out_image > x2).float()
                '''


                #out_image= cv2.threshold(out_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                #val_num_correct += (out_image == segment_image).sum()
                #val_num_pixels += torch.numel(out_image)

                val_Dice += Dice(out_image, segment_image)
                #val_Iou += Iou(out_image, segment_image)

                val_cnt += 1


            #print(f'{epoch}-{i}-val_loss===>>{val_loss.item()}')
            #print(f'num_pixels===>>{val_num_correct, val_num_pixels}')
            #print(f'{epoch}-{i}-val_acc===>>{val_num_correct / val_num_pixels * 100}')

            print(f'{epoch}-{i}-val_Dice===>>{val_Dice / val_cnt}')
            #print(f'{epoch}-{i}-val_IOU===>>{val_Iou / val_cnt}')
            if val_Dice / val_cnt > max_dice:
                max_dice = val_Dice / val_cnt
                max_epoch = epoch
                print('--------------------max_dice=', max_dice)
                print('--------------------max_epoch=', max_epoch)

                torch.save(net.state_dict(), weight_path)
                print('save successfully!')

        epoch += 1






