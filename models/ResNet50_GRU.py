import torchvision.models as models
import torch
import torch.nn as nn
import copy
import numpy as np

def printLearnableParameters(model):
    print("learnable parameter for the Encoder as follows (if any):")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

def loadCheckpoint(encoder, Encoder_CheckPoint, num_classes):
    '''
        Encoder_CheckPoint = dict_keys(['best_epoch_num', 'best_model_wts', 'best_optimizer_wts', 'best_val_acc'])
    '''
    num_ftrs = encoder.fc.in_features
    encoder.fc = nn.Linear(num_ftrs, num_classes)
    print("Transfering Weights for Encoder...")
    print("best validation accuracy", Encoder_CheckPoint['best_val_acc'])
    encoder.load_state_dict(Encoder_CheckPoint['best_model_wts'])

    return encoder


class ResNet50_GRU(nn.Module):
    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def __init__(self, num_classes=4, pretrained=True, resnet50=True,
                 feature_extract=True,Encoder_CheckPoint=None):
        super(ResNet50_GRU, self).__init__()
        if resnet50:
            self.original_ResNet = models.resnet50(pretrained=pretrained)
        else:
            self.original_ResNet = models.resnet101(pretrained=pretrained)
        self.set_parameter_requires_grad(self.original_ResNet,
                                         feature_extract=feature_extract)  # if True freeze all the parameters
        if Encoder_CheckPoint:
            self.original_ResNet = loadCheckpoint(self.original_ResNet,Encoder_CheckPoint, num_classes)
        self.layers = list(self.original_ResNet.children())  # seperate the layers

        # self.layers.insert(9, self.gruUnit)

        self.Encoder = nn.Sequential(*self.layers[:-1])  # combine all layers
        printLearnableParameters(self.Encoder)

        self.gruUnit = nn.GRU(input_size=2048,
                              hidden_size=2048)  # expected input is (seq_len, batch, input_size) (8, 1, 2048)
        self.fc = nn.Sequential(nn.BatchNorm1d(2048),nn.Dropout(p=0.25),nn.Linear(2048, num_classes))  # modify the last fc layer
        # print(self.layers)
        # exit(0)

    def forward(self, x, labels):
        #extract features
        output = self.Encoder(x)  # => (Batch, C, 1, 1) due to the adaptive average pooling
        #split the batch based on the labels
        output_sequences = split_seq_frames(output, labels)
        output_sequences_gru = []
        #for each sub batch do
        for sub_batch in output_sequences:
            sub_batch = sub_batch.squeeze()  # => (Batch, C)
            batch_size = sub_batch.shape[0]

            sub_batch = sub_batch.unsqueeze(1)  # => (batch, 1, C) := (seq, batch, input)
            output, h_n = self.gruUnit(sub_batch)

            h_n = h_n.squeeze()
            h_n = h_n.expand(batch_size,-1)


            output_sequences_gru.append(h_n)
        # ---------------------------------------------------------------------------------

        output = torch.cat(output_sequences_gru)
        output = self.fc(output)

        return output

class ResNet50_max(nn.Module):
    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def __init__(self, num_classes=4, pretrained=True, resnet50=True,
                 feature_extract=True,Encoder_CheckPoint=None):
        super().__init__()
        if resnet50:
            self.original_ResNet = models.resnet50(pretrained=pretrained)
        else:
            self.original_ResNet = models.resnet101(pretrained=pretrained)
        self.set_parameter_requires_grad(self.original_ResNet,
                                         feature_extract=feature_extract)  # if True freeze all the parameters
        if Encoder_CheckPoint:
            self.original_ResNet = loadCheckpoint(self.original_ResNet,Encoder_CheckPoint, num_classes)
        # self.layers = list(self.original_ResNet.children())  # seperate the layers

        # self.layers.insert(9, self.gruUnit)

        self.Encoder = self.original_ResNet  # combine all layers
        printLearnableParameters(self.Encoder)

        # print(self.layers)
        # exit(0)

    def forward(self, x, labels):
        #extract features
        output = self.Encoder(x)  # => (Batch,4) due to the adaptive average pooling

        #split the batch based on the labels
        output_sequences = split_seq_frames(output, labels)
        output_sequences_gru = []
        #for each sub batch do
        for sub_batch in output_sequences:
            sub_batch = sub_batch.squeeze()  # => (Batch, C)
            batch_size = sub_batch.shape[0]
            # print("sub batch=",sub_batch.shape)
            out, indecies = sub_batch.max(dim=0)
            # print(out.shape)
            # out = out.max()
            # print(out)
            # print("result",out)
            h_n = out.expand(batch_size,-1)
            # print("results after expanding",h_n.shape)
            # print(h_n.shape)
            # exit(0)
            output_sequences_gru.append(h_n)
        # ---------------------------------------------------------------------------------

        output = torch.cat(output_sequences_gru)
        # print(output.shape)
        # exit(0)
        # output = self.fc(output)

        return output

class myRNN(nn.Module):
    def __init__(self,input):
        super().__init__()

class ResNet50_SimplerGRU(nn.Module):
    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def __init__(self, num_classes=4, pretrained=True, resnet50=True,
                 feature_extract=True,Encoder_CheckPoint=None):
        super(ResNet50_SimplerGRU, self).__init__()
        if resnet50:
            self.original_ResNet = models.resnet50(pretrained=pretrained)
        else:
            self.original_ResNet = models.resnet101(pretrained=pretrained)
        self.set_parameter_requires_grad(self.original_ResNet,
                                         feature_extract=feature_extract)  # if True freeze all the parameters
        if Encoder_CheckPoint:
            self.original_ResNet = loadCheckpoint(self.original_ResNet,Encoder_CheckPoint, num_classes)
        self.layers = list(self.original_ResNet.children())  # seperate the layers

        # self.layers.insert(9, self.gruUnit)

        self.Encoder = nn.Sequential(*self.layers[:-1])  # combine all layers
        printLearnableParameters(self.Encoder)
        # self.my = nn.
        # self.gruUnit = nn.GRU(input_size=2048,
        #                       hidden_size=2048)  # expected input is (seq_len, batch, input_size) (8, 1, 2048)
        # self.fc = nn.Sequential(nn.BatchNorm1d(2048),nn.Dropout(p=0.25),nn.Linear(2048, num_classes))  # modify the last fc layer
        # print(self.layers)
        # exit(0)

    def forward(self, x, labels):
        #extract features
        output = self.Encoder(x)  # => (Batch, C, 1, 1) due to the adaptive average pooling
        #split the batch based on the labels
        output_sequences = split_seq_frames(output, labels)
        output_sequences_gru = []
        #for each sub batch do
        for sub_batch in output_sequences:
            sub_batch = sub_batch.squeeze()  # => (Batch, C)
            batch_size = sub_batch.shape[0]

            sub_batch = sub_batch.unsqueeze(1)  # => (batch, 1, C) := (seq, batch, input)
            output, h_n = self.gruUnit(sub_batch)

            h_n = h_n.squeeze()
            h_n = h_n.expand(batch_size,-1)


            output_sequences_gru.append(h_n)
        # ---------------------------------------------------------------------------------

        output = torch.cat(output_sequences_gru)
        output = self.fc(output)

        return output

class ResNet50_h_initialized_GRU(nn.Module):
    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def __init__(self, num_classes=4, pretrained=True, resnet50=True, feature_extract=True):
        super(ResNet50_h_initialized_GRU, self).__init__()
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
        output_sequences = split_seq_frames(output, labels)

        output_sequences_gru = []
        for sub_batch in output_sequences:
            output, hidden = self.gruUnit(sub_batch)
            output = output.squeeze()
            output_sequences_gru.append(output)
        # ---------------------------------------------------------------------------------
        output = torch.cat(output_sequences_gru)
        # print(output.shape)
        output = self.fc(output)
        return output

def split_seq_frames(x, labels):
    ''' I need to split the input into sequences based on the labels
    :parameter
        x: shape is (batch,1, C)
        labels: shape is (batch)
    :return
        array of tensors'''
    # print(x.shape)
    split_indecies = []
    count = 0
    temp = labels[0].item()
    for i in range(len(x)):  # for each label
        frame_label = labels[i].item()

        if temp != frame_label:  # a potential split is at this index
            split_indecies.append(count)
            count = 0
            temp = frame_label
        count += 1
    split_indecies.append(count)

    if len(split_indecies) == 1:  # if the labels are homogeneous (batch has only one label)
        return (x,)
    # print(split_indecies)
    # # del(split_indecies[0])#the first index is always 0 since and we dont want to split the begining
    # # split_indecies.append(len(labels)-np.sum(split_indecies))
    # print(labels)

    # print(torch.split(labels,split_indecies))
    # exit(0)
    # print(torch.split(x,split_indecies)[0].shape)
    # print(torch.split(x,split_indecies)[1].shape)
    return torch.split(x, split_indecies, dim=0)
