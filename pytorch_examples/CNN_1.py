# CNN_1.py
# editor: tongpl
# data: 2020.3.9

import torch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio


# 学习率
LR = 0.0005

class MyRelu:
    """
    http://gmei test
    """
    
    def __init__(self):
        self.x = None

    def __call__(self,x):
        self.x = x
        #must use copy in nump to avaoid pass by reference
        out = slef.x.copy()
        out[out<0] = 0 
        return out

    def backward(self, d_loss):
        relu_mask = (self.x >= 0)
        dx = d_loss * relu_mask 
        return dx

##class Net(torch.nn.Module):
##    """Some Information about Net"""
##
##    def __init__(self):
##        super(Net, self).__init__()
##        # sequential: 时序容器，以他们传入的顺序被添加到容器中
##        m=torch.nn.ReLU();
##        self.conv1 = torch.nn.Sequential(
##            torch.nn.Conv2d(3, 16, 3, padding=1),  # 3*32*32 -> 16*32*32
##            MyRelu(),
##            #torch.nn.ReLU(),
##            torch.nn.MaxPool2d(2, 2)  # 16*32*32 -> 16*16*16
##        )
##        self.conv2 = torch.nn.Sequential(
##            torch.nn.Conv2d(16, 32, 3, padding=1),  # 16*16*16 -> 32*16*16
##            MyRelu(),
##            #torch.nn.ReLU(),
##            torch.nn.MaxPool2d(2, 2)  # 32*16*16 -> 32*8*8
##        )
##        self.conv3 = torch.nn.Sequential(
##            torch.nn.Conv2d(32, 64, 3, padding=1),  # 32*8*8 -> 64*8*8
##            MyRelu(),
##            #torch.nn.ReLU(),
##            torch.nn.MaxPool2d(2, 2)  # 64*8*8 -> 64*4*4
##        )
##        self.fc1 = torch.nn.Sequential(
##            torch.nn.Linear(64*4*4, 32),
##            MyRelu(),
##            #torch.nn.ReLU(),
##            # torch.nn.Dropout()
##        )
##        self.fc2 = torch.nn.Linear(32, 10)
##
##    def forward(self, x):
##        x = self.conv1(x)
##        x = self.conv2(x)
##        x = self.conv3(x)
##        x = x.view(-1, 64*4*4)
##        x = self.fc1(x)
##        x = self.fc2(x)
##        return x
##

class Net1(torch.nn.Module):
    """Some Information about CNN"""

    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),  # 3*32*32 -> 16*32*32
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, padding=1),  # 16*32*32 -> 32*32*32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 32*32*32-> 32*16*16
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, padding=1),  #  32*16*16 -> 64*16*16
            torch.nn.ReLU(),
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1),  #  64*16*16 -> 128*16*16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 128*16*16 -> 128*8*8
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, padding=1),  #  128*8*8 -> 256*8*8
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 256*8*8 -> 256*4*4
        )

        ##self.gap = torch.nn.AvgPool2d(4,4)
        self.gap = torch.nn.MaxPool2d(4,4)
        self.fc = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        ##print('cov1 out shape',x.shape);
        ##plt.figure(1)
        ##for i in range(1, 16 + 1):
        ##    plt.subplot(4, 4, i)
        ##    plt.imshow(x[0,i-1])
        x = self.conv2(x)
        ##print('cov2 out shape',x.shape);
        ##plt.figure(2)
        ##for i in range(1, 32 + 1):
        ##    plt.subplot(4, 8, i)
        ##    plt.imshow(x[0, i - 1])
        x = self.conv3(x)
        ##print('cov3 out shape',x.shape);
        ##plt.figure(3)
        ##for i in range(1, 64 + 1):
        ##    plt.subplot(4, 16, i)
        ##    plt.imshow(x[0, i - 1])
        x = self.conv4(x)
        ##print('cov4 out shape',x.shape);
        ##plt.figure(4)
        ##for i in range(1, 128 + 1):
        ##    plt.subplot(8, 16, i)
        ##    plt.imshow(x[0, i - 1])
        x = self.conv5(x)
        ##print('cov5 out shape',x.shape);
        x = self.gap(x)
        ##print('gap out shape',x.shape);
        ##plt.show()
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

##class Net2(torch.nn.Module):
##    """Some Information about Net"""
##
##    def __init__(self):
##        super(Net2, self).__init__()
##        # sequential: 时序容器，以他们传入的顺序被添加到容器中
##        self.conv1 = torch.nn.Sequential(
##            torch.nn.Conv2d(3, 31, 3, padding=1),  # 3*32*32 -> 31*32*32
##            torch.nn.ReLU()
##        )
##        self.conv2 = torch.nn.Sequential(
##            torch.nn.Conv2d(31, 32, 3, padding=1),  # 31*32*32 -> 32*32*32
##            torch.nn.ReLU()
##        )
##        self.conv3 = torch.nn.Sequential(
##            torch.nn.Conv2d(32, 33, 3, padding=1),  # 32*32*32 -> 33*32*32
##            torch.nn.ReLU(),
##            torch.nn.MaxPool2d(2, 2)  # 64*32*32 -> 64*16*16
##        )
##        self.conv4 = torch.nn.Sequential(
##            torch.nn.Conv2d(33, 64, 3, padding=1),  # 33*16*16 -> 64*16*16
##            torch.nn.ReLU()
##        )
##        self.conv5 = torch.nn.Sequential(
##            torch.nn.Conv2d(64, 65, 3, padding=1),  # 64*16*16 -> 65*16*16
##            torch.nn.ReLU(),
##            torch.nn.MaxPool2d(2, 2)  # 65*16*16 -> 65*8*8
##        )
##
##        self.fc1 = torch.nn.Sequential(
##            torch.nn.Linear(65*8*8, 10),
##            torch.nn.Softmax(dim=1),
##            # torch.nn.Dropout()
##        )
##
##    def forward(self, x):
##        x = self.conv1(x)
##        x = self.conv2(x)
##        x = self.conv3(x)
##        x = self.conv4(x)
##        x = self.conv5(x)
##        x = x.view(-1, 65*8*8)
##        x = self.fc1(x)
##        return x
##

net = Net1()
#net.cuda()
net.cpu()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

