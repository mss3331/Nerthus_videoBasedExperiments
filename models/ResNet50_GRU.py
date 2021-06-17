import torchvision.models as models
import torch
import torch.nn as nn
class ResNet50_GRU(nn.Module):
    def set_parameter_requires_grad(self,model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def __init__(self,num_classes=4,pretrained=True, resnet50=True, feature_extract =True):
        super(ResNet50_GRU, self).__init__()
        if resnet50:
            self.original_ResNet = models.resnet50(pretrained=pretrained)
        else:
            self.original_ResNet = models.resnet101(pretrained=pretrained)
        self.set_parameter_requires_grad(self.original_ResNet, feature_extract=feature_extract) #if True freeze all the parameters


        self.layers =list(self.original_ResNet.children()) # seperate the layers



        # self.layers.insert(9, self.gruUnit)

        self.Encoder = nn.Sequential(*self.layers[:-1])  # combine all layers
        self.gruUnit = nn.GRU(input_size=2048, hidden_size=2048)  # expected input is (seq_len, batch, input_size) (8, 1, 2048)
        self.fc = nn.Linear(2048, num_classes)  # modify the last fc layer
        # print(self.layers)
        # exit(0)


    def forward(self, x, labels):
        output = self.Encoder(x) #=> (Batch, C, 1, 1) due to the adaptive average pooling
        # print(x.shape)
        output= output.squeeze() #=> (Batch, C)
        output = output.unsqueeze(1) #=> (batch, 1, C) := (seq, batch, input)

        output, hidden = self.gruUnit(output)
        output = output.squeeze()
        print(output.shape)
        output = self.fc(output)
        return output

class ResNet50_h_initialized_GRU(nn.Module):
    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def __init__(self, num_classes=4, pretrained=True, resnet50=True, feature_extract=True):
        super(ResNet50_GRU, self).__init__()
        if resnet50:
            self.original_ResNet = models.resnet50(pretrained=pretrained)
        else:
            self.original_ResNet = models.resnet101(pretrained=pretrained)
        self.set_parameter_requires_grad(self.original_ResNet,
                                         feature_extract=feature_extract)  # if True freeze all the parameters

        self.layers = list(self.original_ResNet.children())  # seperate the layers

        # self.layers.insert(9, self.gruUnit)

        self.Encoder = nn.Sequential(*self.layers[:-1])  # combine all layers
        self.gruUnit = nn.GRU(input_size=2048,
                              hidden_size=2048)  # expected input is (seq_len, batch, input_size) (8, 1, 2048)
        self.fc = nn.Linear(2048, num_classes)  # modify the last fc layer
        # print(self.layers)
        # exit(0)

    def forward(self, x, labels):
        output = self.Encoder(x)  # => (Batch, C, 1, 1) due to the adaptive average pooling
        # print(x.shape)
        output = output.squeeze()  # => (Batch, C)
        output = output.unsqueeze(1)  # => (batch, 1, C) := (seq, batch, input)

        # ------------------------------------------------------------------------------
        # we want here to split the batch into seperate sequences based on the class label
        output_sequences = self.split_seq_frames(output, labels)
        output_sequences_gru = []
        for sub_batch in output_sequences:
            output, hidden = self.gruUnit(sub_batch)
            output = output.squeeze()
            output_sequences_gru.append(output)
        # ---------------------------------------------------------------------------------
        output = torch.cat(output_sequences_gru)
        print(output.shape)
        output = self.fc(output)
        return output

    def split_seq_frames(self, x, labels):
        ''' I need to split the input into sequences based on the labels
        :parameter
            x: shape is (batch,1, C)
            labels: shape is (batch)
        :return
            array of tensors'''
        # print(x.shape)
        split_indecies = []
        temp = -1
        for i in range(len(x)): # for each label
            frame_label = labels[i].item()
            if temp != frame_label: #a potential split is at this index
                split_indecies.append(i)
                temp = frame_label
        del(split_indecies[0])#the first index is always 0 since and we dont want to split the beginin
        split_indecies.append(len(labels)-split_indecies[-1])
        # print(labels)
        # print(torch.split(x,split_indecies)[0].shape)
        # print(torch.split(x,split_indecies)[1].shape)
        return torch.split(x,split_indecies,dim=0)
