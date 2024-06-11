import torch
from torch import nn
from torch.nn import functional as F



class Threshold(nn.Module):
    def __init__(self):
        super(Threshold, self).__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(65536,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 0, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.1),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)


class UNet(nn.Module):
    #def __init__(self,num_classes):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1=Conv_Block(1,32)
        self.d1=DownSample(32)
        self.c2=Conv_Block(32,64)
        self.d2=DownSample(64)
        self.c3=Conv_Block(64,128)
        self.d3=DownSample(128)
        self.c4=Conv_Block(128,256)
        self.d4=DownSample(512)
        self.c5=Conv_Block(512,1024)
        self.u1=UpSample(1024)
        self.c6=Conv_Block(1024,512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        #self.out=nn.Conv2d(64,num_classes,3,1,1)
        self.out = nn.Conv2d(64, 1, 3, 1, 1)

        self.tres=Threshold()
        self.Th=nn.Sigmoid()

        self.se1 = doubleSE_Block(32)
        self.se2 = doubleSE_Block(64)
        self.se3 = doubleSE_Block(128)
        self.se4 = doubleSE_Block(256)


    def forward(self,x):
        x1=x[:,0,:,:]
        x1=x1.unsqueeze(1)
        x2 = x[:, 1, :, :]
        x2 = x2.unsqueeze(1)

        R11=  self.c1(x1)
        R12 = self.c1(x2)
        R1 = self.se1(R11,R12)

        R21 = self.c2(self.d1(R11))
        R22 = self.c2(self.d1(R12))
        R2 = self.se2(R21, R22)

        R31 = self.c3(self.d2(R21))
        R32 = self.c3(self.d2(R22))
        R3 = self.se3(R31, R32)


        R41 = self.c4(self.d3(R31))
        R42 = self.c4(self.d3(R32))
        R4 = self.se4(R41, R42)

        R5 = self.c5(self.d4(R4))

        O1=self.c6(self.u1(R5,R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        #res=self.tres(x)

        out1 = self.Th(self.out(O4))
        return  out1


class SE_Block(nn.Module):                         # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y= x
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)*y
        return out

class doubleSE_Block(nn.Module):  # Squeeze-and-Excitation block
    def __init__(self,in_planes):
        super(doubleSE_Block, self).__init__()
        self.seBlock=SE_Block(in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2):
        y1=x1
        y2=x2
        x1 = self.seBlock(x1)
        x2 = self.seBlock(x2)

        out=x1+x2
        out1 = self.sigmoid(y1) * out
        out2 = self.sigmoid(y2) * out
        out = torch.cat([out1, out2], dim=1)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    x=torch.randn(4,2,256,256).to(device)
    #net=UNet(3)
    net=UNet().cuda()
    out=net(x)
    print(out.shape)


 
