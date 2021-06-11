import torchvision.models as models
import torch
import torch.nn as nn
class ResNet50_GRU(nn.Module):
    def set_parameter_requires_grad(self,model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def __init__(self,num_classes=4,pretrained=True):
        super(ResNet50_GRU, self).__init__()
        self.original_ResNet50 = models.resnet50(pretrained=pretrained)
        self.set_parameter_requires_grad(self.original_ResNet50, feature_extracting=True) # freeze all the parameters


        self.layers =list(self.original_ResNet50.children()) # seperate the layers



        # self.layers.insert(9, self.gruUnit)

        self.Encoder = nn.Sequential(*self.layers[:-1])  # combine all layers
        self.gruUnit = nn.GRU(input_size=2048, hidden_size=2048)  # expected input is (seq_len, batch, input_size) (8, 1, 2048)
        self.fc = nn.Linear(2048, num_classes)  # modify the last fc layer
        # print(self.layers)
        # exit(0)


    def forward(self, x):
        output = self.Encoder(x)
        output= output.squeeze()
        output = output.unsqueeze(1)
        output, hidden = self.gruUnit(output)
        output = output.squeeze()
        output = self.fc(output)
        return output