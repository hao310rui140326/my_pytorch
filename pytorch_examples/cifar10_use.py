#author: tpl
#date: 2019.12.18
#Classifier use PyTorch (CIFAR10 dataset)

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import  Variable
import torch.nn as nn
import torch.nn.functional as F
import CNN_1
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio

import sys
sys.path.append("../Pytorch-LRP-master")
print (sys.path)
from innvestigator import InnvestigateModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置训练参数
EPOCHS = 10
BATCH_SIZE = 100
SET_SIZE = 50000
TBATCH_SIZE = 10
TSTEP_SIZE  = 10
LBATCH_SIZE = 5
LSTEP_SIZE  = 5

# 创建一个转换器，将torchvision数据集的输出范围[0,1]转换为归一化范围的张量[-1,1]。
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

##help(torchvision)

# 创建训练集
trainset = torchvision.datasets.CIFAR10(root='../../cifar10_dataset', train=True,
                                        download=True, transform=transform)
# 创建训练加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=1)

# 创建测试集
testset = torchvision.datasets.CIFAR10(root='../../cifar10_dataset', train=False,
                                       download=True, transform=transform)
# 创建测试加载器
testloader = torch.utils.data.DataLoader(testset, batch_size=TBATCH_SIZE,
                                         shuffle=False, num_workers=1)

# 创建测试加载器
lrploader = torch.utils.data.DataLoader(testset, batch_size=LBATCH_SIZE,
                                         shuffle=False, num_workers=1)

# 类别标签
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(model, criterion, optimizer, trainloader, epochs=5, log_interval=50):
    print('----- Train Start -----')
    for epoch in range(epochs):
        running_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(trainloader):
            if step < SET_SIZE/BATCH_SIZE:
                # get inputs
                ##batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                output = model(batch_x)

                optimizer.zero_grad()
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if step % log_interval == (log_interval-1):
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, step + 1, running_loss / log_interval))
                    running_loss = 0.0
    print('----- Train Finished -----')
    
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())

    torch.save(model.state_dict(), 'model.pth')
    print('----- Save Finished -----')


def test(model, testloader):
    print('------ Test Start -----')

    correct = 0
    total = 0
    i=0

    with torch.no_grad():
        for step, (test_x, test_y) in enumerate(testloader):
            if step < TSTEP_SIZE:
                ##images, labels = test_x.cuda(), test_y.cuda()
                images, labels = test_x.cpu(), test_y.cpu()
                output = model(images)
                ##print('output',output)
                #plt.figure(step)
                plt.figure(0)
                for i in range(1,TBATCH_SIZE+1):
                    plt.subplot(TSTEP_SIZE, TBATCH_SIZE, i+step*TBATCH_SIZE)
                    my_imshow(images[i-1])
                _, predicted = torch.max(output.data, 1)
                print('step',step,' ',end='')
                print('predicted',' ',end='')
                for i in range(1,TBATCH_SIZE+1):
                    index=predicted[i-1]
                    print(classes[index],' ',end='')
                print('')
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('images.shape',images.shape)
    print('output.shape',output.shape)
    print('Accuracy of the network is: %.4f %%' % accuracy)
    plt.show()
    return accuracy

def test_all(model, testloader):
    print('------ Test all Start -----')

    correct = 0
    total = 0
    i=0

    with torch.no_grad():
        for step, (test_x, test_y) in enumerate(testloader):
            images, labels = test_x.cpu(), test_y.cpu()
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('step'+' is '+str(step)+' '+'total'+' is '+str(total)+' '+'correct'+' is '+str(correct)+' ')
    accuracy = 100 * correct / total
    return accuracy    

def my_imshow(img):
    """ Tiny helper to show images as uint8 and remove axis labels """
    ##kitten, puppy = imageio.imread('kitten.jpg'), imageio.imread('puppy.jpg')
    ##print('kitten.shape', kitten.shape)
    ##print('puppy.shape', puppy.shape)
    ##imshow_noax(kitten, normalize=False)
    my_imge = np.zeros((32, 32, 3))
    my_imge = np.array(img).transpose((1, 2, 0))
    imshow_noax(my_imge, normalize=True)

def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')

def main_func(argin):
    if argin == 1:
        # Train the network
        train(CNN_1.net, CNN_1.criterion, CNN_1.optimizer, trainloader, epochs=EPOCHS)
        torch.save(CNN_1.net.state_dict(),'model.pth')
    elif argin==2:
        # Test the network
        CNN_1.net.load_state_dict(torch.load('model.pth'))
        CNN_1.net.eval()
        test_all(CNN_1.net, testloader)
    elif argin==3:
        # Test the network
        CNN_1.net.load_state_dict(torch.load('model.pth'))
        CNN_1.net.eval()
        test(CNN_1.net, testloader)    
    elif argin==4:    
        # Get LRP result
        # Convert to innvestigate model
        CNN_1.net.load_state_dict(torch.load('model.pth'))
        inn_model = InnvestigateModel(CNN_1.net, lrp_exponent=2,
                                  method="e-rule",
                                  beta=.5)                 
        with torch.no_grad():
            for step, (test_x, test_y) in enumerate(lrploader):
                if step < LSTEP_SIZE :
                    images, labels = test_x.cpu(), test_y.cpu()
                    output = CNN_1.net(images)
                    ##plt.imshow(images[0])
                    ##print('images.shape',images.shape)
                    ##images = images[0].reshape(-1, 32*32*3).to(device)
                    model_prediction, heatmap = inn_model.innvestigate(in_tensor=images)
                    ##print(model_prediction)
                    ##print('heatmap.shape',heatmap.shape)
                    ##print(heatmap.shape)
                    for i in range(1, LBATCH_SIZE + 1):
                        plt.subplot(LSTEP_SIZE, LBATCH_SIZE*2, i*2 -1  + step * LBATCH_SIZE * 2)
                        my_imshow(images[i - 1])
                        plt.subplot(LSTEP_SIZE, LBATCH_SIZE * 2, i*2  + step * LBATCH_SIZE * 2)
                        my_imshow(heatmap[i - 1])
        print('images.shape', images.shape)
        print('heatmap.shape', heatmap.shape)
        plt.show()

if __name__ ==  '__main__':
    main_func(4)