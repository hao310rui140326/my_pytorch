import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../Pytorch-LRP-master")
print (sys.path)
from innvestigator import InnvestigateModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters - original
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Hyper-parameters - LRP
lrp_input_size = 784
lrp_hidden_size = 500
lrp_num_classes = 10
lrp_num_epochs = 5
lrp_batch_size = 5
lrp_learning_rate = 0.001


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


# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../mnist_data',
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../mnist_data',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

lrp_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=lrp_batch_size, 
                                          shuffle=False)
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)


model.load_state_dict(torch.load('./minst.pt'))
# Convert to innvestigate model
inn_model = InnvestigateModel(model, lrp_exponent=2,
                              method="e-rule",
                              beta=.5)


for epoch in range(lrp_num_epochs):
    for i, (images, labels) in enumerate(lrp_test_loader):  
        if i < lrp_num_epochs:
            images_in = images.reshape(-1, 28*28).to(device)
            model_prediction, heatmap = inn_model.innvestigate(in_tensor=images_in)
            
            heatmap = heatmap.reshape(lrp_batch_size,28,28)
            for step in range(1, lrp_batch_size + 1):
                plt.subplot(lrp_num_epochs, lrp_batch_size*2, step*2 -1  + i * lrp_batch_size * 2)
                plt.imshow(images[step - 1,0])
                plt.subplot(lrp_num_epochs,  lrp_batch_size* 2, step*2  + i * lrp_batch_size * 2)
                plt.imshow(heatmap[step - 1])

print('heatmap shape ',heatmap.shape)
print('image shape ',images.shape)
plt.show()



