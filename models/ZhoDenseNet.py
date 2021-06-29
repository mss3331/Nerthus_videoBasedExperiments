import torchvision.models as models
import torch.nn as nn
import torch

class ZhoDenseNet(nn.Module):
    def __init__(self,num_classes,pretrained=True):
        super(ZhoDenseNet, self).__init__()
        in_channels, self.block1 = self.Block(3,32,3)
        in_channels, self.block2 = self.Block(in_channels,64,2)
        in_channels, self.block3 = self.Block(in_channels,128,2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(56448, 1024), nn.ReLU(), nn.BatchNorm1d(1024))
        self.fc2 = nn.Sequential(nn.Dropout(0.5),nn.Linear(1024, 4))



    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = nn.functional.softmax(x)
        return x


    def Block(self,in_channels, out_channels,pool_kern):
        return out_channels, nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
                             nn.ReLU(),
                             nn.BatchNorm2d(out_channels),
                             nn.MaxPool2d(pool_kern),
                             nn.Dropout(0.25))

class ZhoDenseNet2(nn.Module):
    def __init__(self,num_classes,pretrained=True):
        super(ZhoDenseNet2, self).__init__()
        in_channels, self.block1 = self.Block(3,32,3)
        in_channels, self.block2 = self.Block(in_channels,64,2)
        in_channels, self.block3 = self.Block(in_channels,128,2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(56448, 1024), nn.ReLU(), nn.BatchNorm1d(1024))
        self.fc2 = nn.Sequential(nn.Dropout(0.5),nn.Linear(57472, 4))



    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x1 = self.fc1(x)
        # print(x1.shape)
        x = torch.cat((x1,x),dim=1)
        # print(x.shape)
        x = self.fc2(x)
        # x = nn.functional.softmax(x)
        return x


    def Block(self,in_channels, out_channels,pool_kern):
        return out_channels, nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
                             nn.ReLU(),
                             nn.BatchNorm2d(out_channels),
                             nn.MaxPool2d(pool_kern),
                             nn.Dropout(0.25))