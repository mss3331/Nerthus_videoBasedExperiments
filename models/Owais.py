# this model are implemented based on the following paper:Artificial Intelligence-Based Classification
# of Multiple Gastrointestinal Diseases Using Endoscopy Videos for Clinical Diagnosis by Muhammad Owais
# It is initially tested on Kvasir dataset and Gastrolab videos
import torchvision.models as models
import torch
import torch.nn as nn

class Owais_ResNet18_LSTM(nn.Module):
    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def __init__(self, num_classes=4, pretrained=True, feature_extract=True):
        super(Owais_ResNet18_LSTM, self).__init__()
        self.original_ResNet = models.resnet18(pretrained=pretrained)
        self.set_parameter_requires_grad(self.original_ResNet,
                                         feature_extract=feature_extract)  # if True freeze all the parameters

        self.layers = list(self.original_ResNet.children())  # seperate the layers

        # self.layers.insert(9, self.gruUnit)

        self.Encoder = nn.Sequential(*self.layers[:-1])  # combine all layers
        self.lstmUnit = nn.LSTM(input_size=512,
                              hidden_size=600)  # expected input is (seq_len, batch, input_size) (8, 1, 2048)
        self.fc = nn.Sequential(nn.Dropout(p=0.1),nn.Linear(600, num_classes))  # modify the last fc layer
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
            output, (h_n, c_n) = self.lstmUnit(sub_batch)

            h_n = h_n.squeeze()
            h_n = h_n.expand(batch_size,-1)


            output_sequences_gru.append(h_n)
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