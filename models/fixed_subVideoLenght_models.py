import torchvision.models as models
import torch
import torch.nn as nn
from .MLP_Mixer import MlpBlock, MixerBlock

class Mlp(nn.Module):
    """Entire network.

    Parameters
    ----------
    image_size : int
        Height and width (assuming it is a square) of the input image.

    patch_size : int
        Height and width (assuming it is a square) of the patches. Note
        that we assume that `image_size % patch_size == 0`.

    tokens_mlp_dim : int
        Hidden dimension for the `MlpBlock` when doing the token mixing.

    channels_mlp_dim : int
        Hidden dimension for the `MlpBlock` when diong the channel mixing.

    n_classes : int
        Number of classes for classification.

    hidden_dim : int
        Dimensionality of patch embeddings.

    n_blocks : int
        The number of `MixerBlock`s in the architecture.

    Attributes
    ----------
    patch_embedder : nn.Conv2D
        Splits the image up into multiple patches and then embeds each of them
        (using shared weights).

    blocks : nn.ModuleList
        List of `MixerBlock` instances.

    pre_head_norm : nn.LayerNorm
        Layer normalization applied just before the classification head.

    head_classifier : nn.Linear
        The classification head.
    """

    def __init__(self, *,n_frames, encoder_features , tokens_mlp_dim=None, channels_mlp_dim=None,
                 n_blocks=5):
        # number of patches is equivalant to number of images if applied for a video
        super().__init__()
        # n_patches = (image_size // patch_size) ** 2 # assumes divisibility
        self.n_patches = n_frames # this is the maximium number of images for a subvideo
        hidden_dim = encoder_features #number of features that ResNet50 provide
        # No need for embeding since we have an Encoder (ResNet18)
        # self.patch_embedder = nn.Conv2d(
        #     3,
        #     hidden_dim,
        #     kernel_size=patch_size,
        #     stride=patch_size,
        # )

        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    n_patches=self.n_patches,
                    hidden_dim=hidden_dim,
                    tokens_mlp_dim=tokens_mlp_dim,
                    channels_mlp_dim=channels_mlp_dim,
                )
                for _ in range(n_blocks)
            ]
        )

        self.pre_head_norm = nn.LayerNorm(hidden_dim)
        # self.head_classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of square images of shape
            `(n_samples, n_channels, image_size, image_size)`.

        Returns
        -------
        torch.Tensor
            Class logits of shape `(n_samples, n_classes)`.
        """
        # no need for embedder and re arrange since our encoder ResNet18 will produce (Batch, 512, 1, 1)
        # x = self.patch_embedder(x)  # (n_samples, hidden_dim, n_patches ** (1/2), n_patches ** (1/2))
        # x = einops.rearrange(x, "n c h w -> n (h w) c")  # (n_samples, n_patches, hidden_dim)
        #x = (subvideos, frames"vectors", Encoder_out_features)
        for mixer_block in self.blocks:
            x = mixer_block(x)  # (n_samples, n_patches, hidden_dim) is equivlant to (n_subvideos, n_frames, hidden_dim)

        x = self.pre_head_norm(x)  # (n_subvideos, n_frames, hidden_dim)
        x = x.mean(dim=1)  # (n_subvideos, hidden_dim) global average pooling in the original paper

        return x

class ResNet_subVideo_MLP(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_MLP, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)
        # I am expecting 25 vectors for each sub-video. This line needs to be changed if num of images per subvideo change
        self.Mlp = Mlp(n_frames=25, encoder_features= self.encoder_out_features )

        # (vectore from sequence + vector from non-sequence) = encoder_out_features*2
        self.fc = nn.Linear(self.encoder_out_features*2, num_classes)


    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x.shape
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        x_gru = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        x_Mlp = self.Mlp(x)

        # *************** this is default code********************
        x_cat = torch.cat((x_Mlp, x_gru), dim=1)  # -> (subvideos, Encoder_out_features*2)

        output = self.fc(x_cat)  # -> (subvideos, 4)

        return output

class ResNet_subVideo_FcVert(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_FcVert, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)
        # I am expecting 25 vectors for each sub-video. This line needs to be changed if num of images per subvideo change
        self.FcVert = nn.Linear(self.encoder_out_features,1)
        self.normFcVert = nn.BatchNorm1d(25) # I am expecting to have one scalar for each frame. a subvideo has 25 frames

        # (vectore from sequence + vector from non-sequence) = encoder_out_features+25
        self.fc = nn.Linear(self.encoder_out_features+25, num_classes)


    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x.shape
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        x_gru = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        #(subvideos, frames"vectors", Encoder_out_features) -> (subvideos*frames"vectors", Encoder_out_features)
        x = x.view((-1,x_shape[-1]))
        #(subvideos * frames"vectors", Encoder_out_features) -> (subvideos*frames"vectors", 1)
        # ->(subvideos*frames"vectors") -> (subvideos, frames"vectors")
        x_fc = self.FcVert(x).squeeze().view((x_shape[0],x_shape[1]))
        x_fc = self.normFcVert(x_fc)

        # *************** this is default code********************
        x_cat = torch.cat((x_fc, x_gru), dim=1)  # -> (subvideos, Encoder_out_features+25)

        output = self.fc(x_cat)  # -> (subvideos, 4)

        return output

class ResNet_subVideo_FcHoriz(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_FcHoriz, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)
        # I am expecting 25 vectors for each sub-video. This line needs to be changed if num of images per subvideo change
        self.FcHoriz = nn.Linear(25,1)
        self.normFcHoriz = nn.BatchNorm1d(self.encoder_out_features)

        # (vectore from sequence + vector from non-sequence) = encoder_out_features*2
        self.fc = nn.Linear(self.encoder_out_features*2, num_classes)


    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x.shape
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        x_gru = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        # (subvideos, frames"vectors", Encoder_out_features) -> (subvideos, Encoder_out_features, frames"vectors")
        x = x.permute((0,2,1))
        # (subvideos, Encoder_out_features, frames"vectors") -> (subvideos*Encoder_out_features, frames"vectors")
        x = x.reshape((-1,x_shape[1])) # I used reshape instead of view due to an error (contigous memory)
        # (subvideos*Encoder_out_features, frames"vectors") -> (subvideos*Encoder_out_features, 1 vector)
        x_fc = self.FcHoriz(x)
        # (subvideos * Encoder_out_features, 1 vector) -> (subvideos*Encoder_out_features)
        # ->(subvideos,Encoder_out_features)
        x_fc = x_fc.squeeze().view((x_shape[0],-1))
        x_fc = self.normFcHoriz(x_fc)

        # *************** this is default code********************
        x_cat = torch.cat((x_fc, x_gru), dim=1)  # -> (subvideos, 2*Encoder_out_features)

        output = self.fc(x_cat)  # -> (subvideos, 4)

        return output

class ResNet_subVideo_MaxOnly(nn.Module):#first proposal
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_MaxOnly, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder_WithoutGRU(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normMax = nn.BatchNorm1d(self.encoder_out_features)

        # self.drop = nn.Dropout(p=0.5)
        # (vectore from sequence + vector from non-sequence) = encoder_out_features*2
        self.fc = nn.Linear(self.encoder_out_features, num_classes)


    def forward(self, x):
        #*************this is default code*************
        x_shape = x.shape
        # x=(subvideos, frames"vectors", Encoder_out_features)
        x = self.SubVideo_Encoder(x)

        #************ your non-sequence code ********************
        # (subvideos, frames"vectors", Encoder_out_features) -> (subvideos, Encoder_out_features)
        x_max, _ = x.max(dim=1)
        x_max = self.normMax(x_max)
        #*************** this is default code*******************
        output = self.fc(x_max) # -> (subvideos, 4)

        return output

class ResNet_subVideo_GRU(nn.Module):#first proposal
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_GRU, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)
        # self.drop = nn.Dropout(p=0.5)
        # (vectore from sequence + vector from non-sequence) = encoder_out_features*2
        self.fc = nn.Linear(self.encoder_out_features, num_classes)


    def forward(self, x):
        #*************this is default code*************
        x_shape = x.shape
        output_dic = self.SubVideo_Encoder(x)
        x = output_dic["x"] # x=(subvideos, frames"vectors", Encoder_out_features)
        x_gru = output_dic["x_gru"] # x_gru = (subvideos, Encoder_out_features)
        x_gru = self.normGRU(x_gru)
        #************ your non-sequence code ********************
        # x_mean, _ = x.max(dim=1) # -> (subvideos, Encoder_out_features)

        #*************** this is default code********************
        # x_cat = torch.cat((x_mean,x_gru), dim=1) # -> (subvideos, 2*Encoder_out_features)
        # x_cat_dropout = self.drop(x_cat)
        output = self.fc(x_gru) # -> (subvideos, 4)

        return output
class ResNet_subVideo_Max(nn.Module):#first proposal
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_Max, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.drop = nn.Dropout(p=0.5)
        # (vectore from sequence + vector from non-sequence) = encoder_out_features*2
        self.fc = nn.Linear(self.encoder_out_features*2, num_classes)


    def forward(self, x):
        #*************this is default code*************
        x_shape = x.shape
        output_dic = self.SubVideo_Encoder(x)
        x = output_dic["x"] # x=(subvideos, frames"vectors", Encoder_out_features)
        x_gru = output_dic["x_gru"] # x_gru = (subvideos, Encoder_out_features)

        #************ your non-sequence code ********************
        x_mean, _ = x.max(dim=1) # -> (subvideos, Encoder_out_features)

        #*************** this is default code********************
        x_cat = torch.cat((x_mean,x_gru), dim=1) # -> (subvideos, 2*Encoder_out_features)
        x_cat_dropout = self.drop(x_cat)
        output = self.fc(x_cat_dropout) # -> (subvideos, 4)

        return output
class ResNet_subVideo_Avg(nn.Module):#first proposal
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_Avg, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.drop = nn.Dropout(p=0.5)
        # (vectore from sequence + vector from non-sequence) = encoder_out_features*2
        self.fc = nn.Linear(self.encoder_out_features*2, num_classes)


    def forward(self, x):
        x_shape = x.shape
        output_dic = self.SubVideo_Encoder(x)
        x = output_dic["x"] # x=(subvideos, frames"vectors", Encoder_out_features)
        x_gru = output_dic["x_gru"] # x_gru = (subvideos, Encoder_out_features)

        x_mean = x.mean(dim=1) # -> (subvideos, Encoder_out_features)

        x_cat = torch.cat((x_mean,x_gru), dim=1) # -> (subvideos, 2*Encoder_out_features)
        x_cat_dropout = self.drop(x_cat)
        output = self.fc(x_cat_dropout) # -> (subvideos, 4)

        return output


class SubVideo_Encoder_WithoutGRU(nn.Module):
    '''The purpose of this Module is:
    1- Convert subvideos into vectors
    2- Pass the vectors to a decoder Module'''

    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def __init__(self, num_classes=4, pretrained=True, resnet50=True,
                 feature_extract=True,Encoder_CheckPoint=None):
        super(SubVideo_Encoder_WithoutGRU, self).__init__()
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

    def forward(self, x):
        #expected input dimention : (subvideos"batch", frames, C, H, W)
        subvideos_num, frames_num, channels, H, W = x.shape
        x_shape = x.shape
        #extract features. Encoder expect the following dim (frames, C, H, W)
        x = x.view((-1,*x_shape[2:])) #-> (subvideo*frames, C, H, W)
        x = self.Encoder(x)  # => (subvideo*frames, Encoder_out_features, 1, 1) due to the adaptive average pooling
        x = x.squeeze() # (subvideo*frames, Encoder_out_features)
        # (subvideo*frames, Encoder_out_features) -> (subvideos, frames"vectors", Encoder_out_features), we have a collection of vectors for each subvideo
        x = x.view((subvideos_num,frames_num,-1))

        # x=(subvideos, frames"vectors", Encoder_out_features)
        return x

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
        x = x.squeeze() # (subvideo*frames, Encoder_out_features)
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
