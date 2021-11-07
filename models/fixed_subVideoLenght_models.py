import torchvision.models as models
import torch
import torch.nn as nn

class ResNet_subVideo_Avg(nn.Module):#first proposal
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_Avg, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.encoder_out_features, num_classes)


    def forward(self, x):
        x_shape = x.shape
        output_dic = self.SubVideo_Encoder(x)
        x = output_dic["x"] # x=(subvideos, frames"vectors", Encoder_out_features)
        x_gru = output_dic["x_gru"] # x_gru = (subvideos, Encoder_out_features)

        x_mean = x.mean(dim=1) # -> (subvideos, 1, Encoder_out_features)
        print('expected output (subvideos, 1, 2048) ', x_mean.shape)
        print('expected output (subvideos, 2048) ', x_gru.shape)
        exit(0)
        x_mean = x_mean.squeeze() # -> (subvideos, Encoder_out_features)
        x_cat = torch.cat((x_mean,x_gru), dim=1) # -> (subvideos, 2*Encoder_out_features)
        x_cat_dropout = self.drop(x_cat)
        output = self.fc(x_cat_dropout) # -> (subvideos, 4)

        return output




class SubVideo_Encoder(nn.Module):
    '''The purpose of this Module is:
    1- Convert subvideos into vectors
    2- Combine the vectors for each subvideo using GRU
    3- Pass the vectors and result of GRU to a decoder Module'''

    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def __init__(self, num_classes=4, pretrained=True, resnet50=True,
                 feature_extract=True,Encoder_CheckPoint=None):
        super(SubVideo_Encoder, self).__init__()
        if resnet50:
            self.original_ResNet = models.resnet50(pretrained=pretrained)
        else:
            self.original_ResNet = models.resnet101(pretrained=pretrained)
        self.set_parameter_requires_grad(self.original_ResNet,
                                         feature_extract=feature_extract)  # if True freeze all the parameters

        # load the Encoder checkpoint if any
        if Encoder_CheckPoint:
            self.original_ResNet = loadCheckpoint(self.original_ResNet,Encoder_CheckPoint, num_classes)
        self.layers = list(self.original_ResNet.children())  # seperate the layers
        self.Encoder_out_features = self.layers[-1].in_features
        self.Encoder = nn.Sequential(*self.layers[:-1])  # combine all layers except the last fc layer

        # expected input is (batch, seq_len, input_size)=(3 subvideos, 25 frames, 2048 features)
        self.trail_gruUnit = nn.GRU(input_size=self.Encoder_out_features,
                              hidden_size=self.Encoder_out_features, batch_first=True)
        # self.fc = nn.Sequential(nn.Linear(2048, num_classes))  # modify the last fc layer

    def forward(self, x):
        #expected input dimention : (subvideos"batch", frames, C, H, W)
        subvideos_num, frames_num, channels, H, W = x.shape
        x_shape = x.shape
        #extract features. Encoder expect the following dim (frames, C, H, W)
        x = x.view((-1,*x_shape[2:])) #-> (subvideo*frames, C, H, W)
        x = self.Encoder(x)  # => (subvideo*frames, Encoder_out_features, 1, 1) due to the adaptive average pooling
        x = x.squeeze(x) # (subvideo*frames, Encoder_out_features)
        x = x.view((subvideos_num,frames_num,-1)) # (subvideo*frames, Encoder_out_features) -> (subvideos, frames"vectors", Encoder_out_features), we have a collection of vectors for each subvideo
        _,x_hn = self.trail_gruUnit(x) #x_hn =(D*num_layers,N,Hout)=(1*1,Batch,Encoder_out_features)
        x_hn = x_hn.view((-1, x_hn.shape[-1]))# -> (Batch, Encoder_out_features) = (subvideos, Encoder_out_features)
        #x_hn will be combined with the output of our proposal work
        output_dic = {"x":x,"x_gru":x_hn}
        # output = self.fc(output)
        # x=(subvideos, frames"vectors", Encoder_out_features) , x_gru = (subvideos, Encoder_out_features)
        return output_dic





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
