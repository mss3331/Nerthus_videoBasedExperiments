import torchvision.models as models
import torch.nn as nn
import torch

class stridedConv(nn.Module):
    def __init__(self,num_classes,pretrained=True):
        super(stridedConv, self).__init__()
        #expected image size is H,W 576,720
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=3,stride=3))#,nn.BatchNorm2d(6)) #H,W 192,240
        self.conv2 = nn.Sequential(nn.BatchNorm2d(6),nn.Conv2d(6,12,kernel_size=3,stride=3))#,nn.BatchNorm2d(12)) #H,W 64,80
        self.conv3 = nn.Sequential(nn.BatchNorm2d(12),nn.Conv2d(12,32,kernel_size=2,stride=2))#,nn.BatchNorm2d(32)) #H,W 32,40
        self.conv4 = nn.Sequential(nn.BatchNorm2d(32),nn.Conv2d(32,64,kernel_size=2,stride=2))#,nn.BatchNorm2d(64)) #H,W 16,20
        self.conv5 = nn.Sequential(nn.BatchNorm2d(64),nn.Conv2d(64,128,kernel_size=2,stride=2))#,nn.BatchNorm2d(128)) #H,W 8,10
        self.conv6 = nn.Sequential(nn.BatchNorm2d(128),nn.Conv2d(128,256,kernel_size=2,stride=2))#,nn.BatchNorm2d(256)) #H,W 4,5
        self.conv7 = nn.Sequential(nn.BatchNorm2d(256),nn.Conv2d(256,512,kernel_size=(2,5),stride=(2,5)))#,nn.BatchNorm2d(512)) #H,W 2,1
        self.conv8 = nn.Sequential(nn.BatchNorm2d(512),nn.Conv2d(512,256,kernel_size=(2,1),stride=(2,1)))#,nn.BatchNorm2d(256)) #H,W 1,1
        self.conv9 = nn.Sequential(nn.BatchNorm2d(256),nn.Conv2d(256,num_classes,kernel_size=1))#,nn.BatchNorm2d(num_classes)) #H,W 1,1
        self.fc = nn.Linear(20480,num_classes)
    def forward(self,x):
        x=self.conv1(x)
        # nn.functional.relu(x,inplace=True)
        x=self.conv2(x)
        # nn.functional.relu(x, inplace=True)
        x=self.conv3(x)
        # nn.functional.relu(x, inplace=True)
        x=self.conv4(x)
        # nn.functional.relu(x, inplace=True)
        # x=self.conv5(x)
        # # nn.functional.relu(x, inplace=True)
        # x=self.conv6(x)
        # # nn.functional.relu(x, inplace=True)
        # x=self.conv7(x)
        # # nn.functional.relu(x, inplace=True)
        # x=self.conv8(x)
        # # nn.functional.relu(x, inplace=True)
        # x=self.conv9(x)
        # # nn.functional.relu(x, inplace=True)

        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

class stridedConv_GRU(nn.Module):
    def __init__(self,num_classes,pretrained=True):
        super(stridedConv_GRU, self).__init__()
        #expected image size is H,W 576,720
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=3,stride=3))#,nn.BatchNorm2d(6)) #H,W 192,240
        self.conv2 = nn.Sequential(nn.BatchNorm2d(6),nn.Conv2d(6,12,kernel_size=3,stride=3))#,nn.BatchNorm2d(12)) #H,W 64,80
        self.conv3 = nn.Sequential(nn.BatchNorm2d(12),nn.Conv2d(12,32,kernel_size=2,stride=2))#,nn.BatchNorm2d(32)) #H,W 32,40
        self.conv4 = nn.Sequential(nn.BatchNorm2d(32),nn.Conv2d(32,64,kernel_size=2,stride=2))#,nn.BatchNorm2d(64)) #H,W 16,20
        self.conv5 = nn.Sequential(nn.BatchNorm2d(64),nn.Conv2d(64,128,kernel_size=2,stride=2))#,nn.BatchNorm2d(128)) #H,W 8,10
        self.conv6 = nn.Sequential(nn.BatchNorm2d(128),nn.Conv2d(128,256,kernel_size=2,stride=2))#,nn.BatchNorm2d(256)) #H,W 4,5
        self.gru1_6 = nn.GRU(input_size=256*4*5, hidden_size=256, num_layers=1)
        # self.conv1_6 = nn.Conv2d(256*2,256,kernel_size=1)
        self.conv7 = nn.Sequential(nn.BatchNorm2d(256),nn.Conv2d(256,512,kernel_size=(2,5),stride=(2,5)))#,nn.BatchNorm2d(512)) #H,W 2,1
        self.conv8 = nn.Sequential(nn.BatchNorm2d(512),nn.Conv2d(512,256,kernel_size=(2,1),stride=(2,1)))#,nn.BatchNorm2d(256)) #H,W 1,1
        self.conv9 = nn.Sequential(nn.BatchNorm2d(256),nn.Conv2d(256,num_classes,kernel_size=1))#,nn.BatchNorm2d(num_classes)) #H,W 1,1
        self.fc = nn.Linear(num_classes,num_classes)
        self.h = None #torch.zeros((1,1,256*4*5))
    def forward(self,x):
        x=self.conv1(x)
        # nn.functional.relu(x,inplace=True)
        x=self.conv2(x)
        # nn.functional.relu(x, inplace=True)
        x=self.conv3(x)
        # nn.functional.relu(x, inplace=True)
        x=self.conv4(x)
        # nn.functional.relu(x, inplace=True)
        x=self.conv5(x)
        # nn.functional.relu(x, inplace=True)
        x=self.conv6(x)
        orig_shape = x.shape
        x_viewed = x.view(x.shape[0],1,-1)
        x_gru, h_n = self.gru1_6(x_viewed,self.h)
        self.h = h_n
        x_gru = x_gru.view(orig_shape[0],orig_shape[1],1,1) #(batch,C,1,1)
        # x_gru = x_gru.unsqueez(0)
        # x_gru = x_gru.unsqueez(2)
        # x_gru = x_gru.unsqueez(3)
        # print(x_gru.shape)
        # print(x.shape)
        x = x*x_gru
        # print(x.shape)
        # exit(0)
        # print(x.shape)
        # print(x_gru.shape)
        # x = torch.cat((x,x_gru),1)
        # print(x.shape)
        # x = self.conv1_6(x)
        # print(x.shape)
        # nn.functional.relu(x, inplace=True)
        x=self.conv7(x)
        # nn.functional.relu(x, inplace=True)
        x=self.conv8(x)
        # nn.functional.relu(x, inplace=True)
        x=self.conv9(x)
        # nn.functional.relu(x, inplace=True)
        print(x.shape)
        x = torch.flatten(x,1)

        x = self.fc(x)
        return x
