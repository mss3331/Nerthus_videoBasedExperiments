import torchvision.models as models
import torch
import torch.nn as nn
class ResNet50_RN(nn.Module):
    def set_parameter_requires_grad(self,model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    def getProject(self,stride=3):
        return nn.Sequential (nn.Flatten(start_dim=2), nn.Conv1d(1024,32,3,padding=1,stride=stride),
                              nn.BatchNorm1d(32), nn.ReLU(),nn.MaxPool1d(2),
                              nn.Dropout(0.25), nn.Flatten(start_dim=1))

    def __init__(self,num_classes=4,pretrained=True, type="mm"):
        super(ResNet50_RN, self).__init__()
        self.original_ResNet50 = models.resnet50(pretrained=pretrained)
        self.set_parameter_requires_grad(self.original_ResNet50, feature_extracting=True)


        self.layers =list(self.original_ResNet50.children())[:-1]
        del (self.layers[7:])
        stride = 1
        if type=="mm": stride=3
        # Encoding
        self.project1 = self.getProject(stride)
        self.project2 = self.getProject(stride)
        # RN
        # self.fc_rn1 = nn.Linear(6272,1568)
        # self.fc_rn2 = nn.Linear(1568,300)
        if type=="mm":# it means do matrix multiplication torch.mm "9834496"
            self.relational_network = _RelationalNetwork_mm(in_features=1115136, out_features=10, num_layers=3)  # e.g. [32, 9834496] ==> [32, 6272//2]
        else:
            self.relational_network = _RelationalNetwork(in_features=6272,out_features=6272//2,num_layers=3) #e.g. [32, 6272] ==> [32, 6272//2]
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
        x2=self.project2(x_featerMaps[1])
        ###############################
                   #RN HERE
        # x_RN = self.fc_rn1(x_cat)
        # x_RN = self.fc_rn2(x_RN)
        x_RN = self.relational_network(x1, x2)
        x_RN = x_RN.unsqueeze(-1) # (32,300)====> (32,300,1) == (batch,seq_len,input)

        ###############################
            #The fully connected layers
        x_RN, (hidden, cell)  = self.lstm(x_RN) #h_n = (1, batch, hidden_size)
        hidden = hidden.squeeze(0)
        output = self.fc1(hidden)
        output = self.fc2(output)
        output = self.fc3(output)
        # print(output.shape)
        # print(x_RN.shape)

        return output

class _RelationalNetwork(nn.Module):
    def mlp(self,in_features,out_features,num_layers):
        layers=[]
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

        return nn.Sequential(*layers), out_features

    def __init__(self,in_features, out_features, num_layers):
        super(_RelationalNetwork, self).__init__()
        # self.in_features = in_features
        # self.num_layers=num_layers
        # self.shrink_rate=shrink_rate
        self.network1, out_features = self.mlp(in_features,out_features,num_layers)

    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), 1)
        return self.network1(x_cat)

class _RelationalNetwork_mm(nn.Module):
    def mlp(self,in_features,out_features,num_layers):
        layers=[]
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

        return nn.Sequential(*layers), out_features

    def __init__(self,in_features, out_features, num_layers):
        super().__init__()
        # self.in_features = in_features
        # self.num_layers=num_layers
        # self.shrink_rate=shrink_rate
        self.network1, out_features = self.mlp(in_features,out_features,num_layers)

    def forward(self, x1, x2):
        #x1 shape (n,1056)
        x1 = x1.unsqueeze(-1)  # => (n,features,1)
        x2 = x2.unsqueeze(1)  # => (n,1,features)
        x = torch.bmm(x1, x2) # (n,1056,1056)

        x = x.flatten(1)
        print(x.shape)
        exit(0)
        # x =
        return self.network1(x)