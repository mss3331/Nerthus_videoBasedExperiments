import torchvision.models as models
import torch
import torch.nn as nn
class ResNet50_FE(nn.Module):
    def set_parameter_requires_grad(self,model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def __init__(self,num_classes=4,pretrained=True):
        super(ResNet50_FE, self).__init__()
        self.original_ResNet50 = models.resnet50(pretrained=pretrained)
        self.set_parameter_requires_grad(self.original_ResNet50, feature_extracting=True)


        self.layers =list(self.original_ResNet50.children())[:-1]
        del (self.layers[7:])
        # Encoding
        self.project1 = nn.Sequential (nn.Conv2d(1024,32,1),nn.BatchNorm2d(32), nn.ReLU(),nn.Dropout(0.25), nn.MaxPool2d(2))
        self.project2 = nn.Sequential (nn.Conv2d(1024,32,1),nn.BatchNorm2d(32), nn.ReLU(),nn.Dropout(0.25), nn.MaxPool2d(2))
        # RN

        #LSTM
        self.lstm = nn.LSTM(input_size=1,hidden_size=300,batch_first=True)
        #FC layers + classification layer
        self.fc1 = nn.Linear(300,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,num_classes)


    def forward(self, x):

        x_featerMaps = []
        for i,layer in enumerate(self.layers):

            if i ==6:
                # print("layer"+str(i)+" composed of:")
                for j,sub_layer in enumerate(layer):
                    # print(sub_layer)
                    if j == 0 or j == 5:
                        x= sub_layer(x)
                        x_featerMaps.append(x)
                    else: x = sub_layer(x)
                    # if j ==0:
                    # print("  after sub_layer" + str(j) + " the shape is ", x.shape)
            else:
                # print("before layer" + str(i) + "the shape is ", x.shape)
                x = layer(x)
                # print("before layer" + str(i) + "the shape is ", x.shape)

        x1=self.project1(x_featerMaps[0])
        x2=self.project1(x_featerMaps[1])
        ###############################
                   #RN HERE
        ###############################
        x_RN = (x1+x2).flatten(1).unsqueeze(-1)
        x_RN, (hidden, cell)  = self.lstm(x_RN) #h_n = (1, batch, hidden_size)
        hidden = hidden.squeeze(0)
        output = self.fc1(hidden)
        output = self.fc2(output)
        output = self.fc3(output)
        # print(output.shape)
        # print(x_RN.shape)

        return output