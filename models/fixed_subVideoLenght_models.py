import torchvision.models as models
import torch
import torch.nn as nn
import helpers
import numpy as np
from .MLP_Mixer import MlpBlock, MixerBlock
#################### MLP #############################
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

class ResNet_subVideo_MLPOnly(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_MLPOnly, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        # self.normGRU = nn.BatchNorm1d(self.encoder_out_features)
        # I am expecting 25 vectors for each sub-video. This line needs to be changed if num of images per subvideo change
        self.Mlp = Mlp(n_frames=25, encoder_features= self.encoder_out_features )

        # (vectore from sequence + vector from non-sequence) = encoder_out_features*2
        self.fc = nn.Linear(self.encoder_out_features, num_classes)


    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x.shape
        # x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        # x_gru = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        x_Mlp = self.Mlp(x)

        # *************** this is default code********************
        # x_cat = torch.cat((x_Mlp, x_gru), dim=1)  # -> (subvideos, Encoder_out_features*2)

        output = self.fc(x_Mlp)  # -> (subvideos, 4)

        return output

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
################### KEYFRAME (dimentioned using fc layers) r+g ################################
class ResNet_subVideo_KeyFrameDimentionedAfterPlus(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_KeyFrameDimentionedAfterPlus, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)
        self.Key = nn.Linear(self.encoder_out_features,1)# the highest key will determine the output vector
        self.normKeyFrame = nn.BatchNorm1d(self.encoder_out_features) # I am expecting to have one vector with size
        # self.relu = nn.ReLU()
        # (vectore from sequence + vector from non-sequence) = encoder_out_features+25
        self.plus_dimentioned = _ApplyFCLayers()
        self.fc = nn.Linear(2, num_classes)


    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x_encoder = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x_encoder.shape
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        g = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        #(subvideos, frames"vectors", Encoder_out_features) ->
        # (subvideos*frames"vectors", Encoder_out_features)
        x = x_encoder.view((-1,x_shape[-1]))

        #(subvideos * frames"vectors", Encoder_out_features) -> (subvideos*frames"vectors", 1)
        x_keys = self.Key(x).squeeze().view((x_shape[0],x_shape[1]))# ->(subvideos*frames"vectors") -> (subvideos, frames"vectors")
        _,indecies = x_keys.max(dim=1) #(subvideos, frames"vectors") -> (subvideos, 1) index should be between 0 and 24
        keyVectors = x_encoder[range(x_encoder.shape[0]),indecies, :] # -> (subvideos,Encoder_out_features)

        r = self.normKeyFrame(keyVectors)
        # x_fc = self.relu(x_fc)

        # *************** this is default code********************
        r_plus_g = r+g  # -> (subvideos, Encoder_out_features*2)
        r_plus_g_dimentioned = self.plus_dimentioned(r_plus_g)
        output = self.fc(r_plus_g_dimentioned)  # -> (subvideos, 4)

        return output
class ResNet_subVideo_KeyFramePlusDimentionedV3(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None,normalization_type="None"):
        super(ResNet_subVideo_KeyFramePlusDimentionedV3, self).__init__()
        # normed can have the following values [0=nothing 1=rNormed 2=allNormed 3=gNormed 4]
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.norm_type = normalization_type.split('_')[-1] # ResNet_subVideo_KeyFramePlusDimentionedV2_None --> None
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)
        self.gru_dimentioned = _FCLayer(self.encoder_out_features, out=2)
        self.Key = nn.Linear(self.encoder_out_features,1)# the highest key will determine the output vector
        self.normKeyFrame = nn.BatchNorm1d(self.encoder_out_features) # I am expecting to have one vector with size
        self.keyFrame_dimentioned = _FCLayer(self.encoder_out_features, out=2)
        # self.relu = nn.ReLU()
        # (vectore from sequence + vector from non-sequence) = encoder_out_features+25
        self.fc = nn.Linear(2, num_classes)

    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x_encoder = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x_encoder.shape
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        g = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        #(subvideos, frames"vectors", Encoder_out_features) ->
        # (subvideos*frames"vectors", Encoder_out_features)
        x = x_encoder.view((-1,x_shape[-1]))

        #(subvideos * frames"vectors", Encoder_out_features) -> (subvideos*frames"vectors", 1)
        x_keys = self.Key(x).squeeze().view((x_shape[0],x_shape[1]))# ->(subvideos*frames"vectors") -> (subvideos, frames"vectors")
        _,indecies = x_keys.max(dim=1) #(subvideos, frames"vectors") -> (subvideos, 1) index should be between 0 and 24
        keyVectors = x_encoder[range(x_encoder.shape[0]),indecies, :] # -> (subvideos,Encoder_out_features)

        r = self.normKeyFrame(keyVectors)
        # x_fc = self.relu(x_fc)

        # **************** further dimentioned the vectors r and g to be of size 2 *********
        r_dimentioned = self.keyFrame_dimentioned(r)
        g_dimentioned = self.gru_dimentioned(g)
        #  *************** Normalization stage **********************#
        if self.norm_type == "None":
            pass
        elif self.norm_type =="R":
            r_dimentioned =torch.linalg.vector_norm(g_dimentioned,dim=1,keepdim=True) * r_dimentioned/torch.linalg.vector_norm(r_dimentioned,dim=1,keepdim=True) # -> (subvideos, Encoder_out_features*2)
        elif self.norm_type =="G":
            g_dimentioned =torch.linalg.vector_norm(r_dimentioned,dim=1,keepdim=True) * g_dimentioned/torch.linalg.vector_norm(g_dimentioned,dim=1,keepdim=True) # -> (subvideos, Encoder_out_features*2)  # -> (subvideos, Encoder_out_features*2)
        elif self.norm_type =="RG":
            r_dimentioned = r_dimentioned/torch.linalg.vector_norm(r_dimentioned,dim=1,keepdim=True)
            g_dimentioned = g_dimentioned/torch.linalg.vector_norm(g_dimentioned,dim=1,keepdim=True)
        else:
            print("You should use a valid normalization, the current normalization is ",self.norm_type)
            exit(0)

        r_plus_g_dimentioned = r_dimentioned + g_dimentioned  # -> (subvideos, Encoder_out_features*2)
        # *************** this is default code********************


        output = self.fc(r_plus_g_dimentioned)  # -> (subvideos, 4)

        return output
class ResNet_subVideo_KeyFramePlusDimentionedV2(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None,normalization_type="None"):
        super(ResNet_subVideo_KeyFramePlusDimentionedV2, self).__init__()
        # normed can have the following values [0=nothing 1=rNormed 2=allNormed 3=gNormed 4]
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.norm_type = normalization_type.split('_')[-1] # ResNet_subVideo_KeyFramePlusDimentionedV2_None --> None
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)
        self.gru_dimentioned = nn.Sequential( _FCLayer(self.encoder_out_features, out=64), _FCLayer(input=64, out=2))
        self.Key = nn.Linear(self.encoder_out_features,1)# the highest key will determine the output vector
        self.normKeyFrame = nn.BatchNorm1d(self.encoder_out_features) # I am expecting to have one vector with size
        self.keyFrame_dimentioned = nn.Sequential( _FCLayer(self.encoder_out_features, out=64), _FCLayer(input=64, out=2))
        # self.relu = nn.ReLU()
        # (vectore from sequence + vector from non-sequence) = encoder_out_features+25
        self.fc = nn.Linear(2, num_classes)

    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x_encoder = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x_encoder.shape
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        g = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        #(subvideos, frames"vectors", Encoder_out_features) ->
        # (subvideos*frames"vectors", Encoder_out_features)
        x = x_encoder.view((-1,x_shape[-1]))

        #(subvideos * frames"vectors", Encoder_out_features) -> (subvideos*frames"vectors", 1)
        x_keys = self.Key(x).squeeze().view((x_shape[0],x_shape[1]))# ->(subvideos*frames"vectors") -> (subvideos, frames"vectors")
        _,indecies = x_keys.max(dim=1) #(subvideos, frames"vectors") -> (subvideos, 1) index should be between 0 and 24
        keyVectors = x_encoder[range(x_encoder.shape[0]),indecies, :] # -> (subvideos,Encoder_out_features)

        r = self.normKeyFrame(keyVectors)
        # x_fc = self.relu(x_fc)

        # **************** further dimentioned the vectors r and g to be of size 2 *********
        r_dimentioned = self.keyFrame_dimentioned(r)
        g_dimentioned = self.gru_dimentioned(g)
        #  *************** Normalization stage **********************#
        if self.norm_type == "None":
            pass
        elif self.norm_type =="R":
            r_dimentioned =torch.linalg.vector_norm(g_dimentioned,dim=1,keepdim=True) * r_dimentioned/torch.linalg.vector_norm(r_dimentioned,dim=1,keepdim=True) # -> (subvideos, Encoder_out_features*2)
        elif self.norm_type =="G":
            g_dimentioned =torch.linalg.vector_norm(r_dimentioned,dim=1,keepdim=True) * g_dimentioned/torch.linalg.vector_norm(g_dimentioned,dim=1,keepdim=True) # -> (subvideos, Encoder_out_features*2)  # -> (subvideos, Encoder_out_features*2)
        elif self.norm_type =="RG":
            r_dimentioned = r_dimentioned/torch.linalg.vector_norm(r_dimentioned,dim=1,keepdim=True)
            g_dimentioned = g_dimentioned/torch.linalg.vector_norm(g_dimentioned,dim=1,keepdim=True)
        else:
            print("You should use a valid normalization, the current normalization is ",self.norm_type)
            exit(0)

        r_plus_g_dimentioned = r_dimentioned + g_dimentioned  # -> (subvideos, Encoder_out_features*2)
        # *************** this is default code********************


        output = self.fc(r_plus_g_dimentioned)  # -> (subvideos, 4)

        return output
class ResNet_subVideo_KeyFramePlusDimentioned(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_KeyFramePlusDimentioned, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)
        self.gru_dimentioned = _ApplyFCLayers()
        self.Key = nn.Linear(self.encoder_out_features,1)# the highest key will determine the output vector
        self.normKeyFrame = nn.BatchNorm1d(self.encoder_out_features) # I am expecting to have one vector with size
        self.keyFrame_dimentioned = _ApplyFCLayers()
        # self.relu = nn.ReLU()
        # (vectore from sequence + vector from non-sequence) = encoder_out_features+25
        self.fc = nn.Linear(2, num_classes)


    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x_encoder = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x_encoder.shape
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        g = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        #(subvideos, frames"vectors", Encoder_out_features) ->
        # (subvideos*frames"vectors", Encoder_out_features)
        x = x_encoder.view((-1,x_shape[-1]))

        #(subvideos * frames"vectors", Encoder_out_features) -> (subvideos*frames"vectors", 1)
        x_keys = self.Key(x).squeeze().view((x_shape[0],x_shape[1]))# ->(subvideos*frames"vectors") -> (subvideos, frames"vectors")
        _,indecies = x_keys.max(dim=1) #(subvideos, frames"vectors") -> (subvideos, 1) index should be between 0 and 24
        keyVectors = x_encoder[range(x_encoder.shape[0]),indecies, :] # -> (subvideos,Encoder_out_features)

        r = self.normKeyFrame(keyVectors)
        # x_fc = self.relu(x_fc)

        # **************** further dimentioned the vectors r and g to be of size 2 *********
        r_dimentioned = self.keyFrame_dimentioned(r)
        g_dimentioned = self.gru_dimentioned(g)
        # *************** this is default code********************
        r_plus_g_dimentioned = r_dimentioned+g_dimentioned  # -> (subvideos, Encoder_out_features*2)

        output = self.fc(r_plus_g_dimentioned)  # -> (subvideos, 4)

        return output

class _ApplyFCLayers(nn.Module):
    '''The expected input is probabily 2048. final_out_size determine what is the target final output size.
    tha ratio deterime the ratio of deduction from one layer into the next.
    E.g., if ratio is 0.5 and input = 2048, then the next layer size should be 2048*0.25=n=512.
    stop creaing fc layers if n == final_out_size.
    for our case having 2048 features, we needs log(2048, base=4) = 5 fc layers to have resutls featurs of size 2'''
    def __init__(self, input=2048,final_out_size=2,ratio=0.25):
        super(_ApplyFCLayers, self).__init__()
        assert ratio<1, "ratio should be in the range (0,1)"

        self.fc_layers_list = self.createFCLayers(input,final_out_size,ratio)

    def forward(self,x):
        for i in range(len(self.fc_layers_list)):
            x = self.fc_layers_list[i](x)
        return x

    def createFCLayers(self,input_size,final_out_size,ratio):
        #input size probabily 2048
        fclayers_list = []
        current_input_size = input_size
        for _ in range(12):# it should be range(log(input,base=1/ratio)) log(2048,base=2) given ratio=0.5
            if current_input_size <= final_out_size:
                break
            output_size = int(current_input_size*ratio) # -> 2048*0.5 = 1048
            fclayers_list.append(_FCLayer(current_input_size,output_size)) # -> current_input_size = 2048, 1048
            current_input_size = output_size
        fclayers_list = nn.ModuleList(fclayers_list)
        return fclayers_list
def _FCLayer(input, out, dropout=0):
    fc_block = nn.Sequential(nn.Linear(input,out),
                             nn.ReLU(),nn.BatchNorm1d(out),
                             nn.Dropout(dropout))
    return fc_block

################### KEYFRAME r+g ################################
class ResNet_subVideo_KeyFramePlus(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_KeyFramePlus, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)

        self.Key = nn.Linear(self.encoder_out_features,1)# the highest key will determine the output vector
        self.normKeyFrame = nn.BatchNorm1d(self.encoder_out_features) # I am expecting to have one vector with size
        # self.relu = nn.ReLU()
        # (vectore from sequence + vector from non-sequence) = encoder_out_features+25
        self.fc = nn.Linear(self.encoder_out_features, num_classes)


    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x_encoder = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x_encoder.shape
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        g = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        #(subvideos, frames"vectors", Encoder_out_features) ->
        # (subvideos*frames"vectors", Encoder_out_features)
        x = x_encoder.view((-1,x_shape[-1]))

        #(subvideos * frames"vectors", Encoder_out_features) -> (subvideos*frames"vectors", 1)
        x_keys = self.Key(x).squeeze().view((x_shape[0],x_shape[1]))# ->(subvideos*frames"vectors") -> (subvideos, frames"vectors")
        _,indecies = x_keys.max(dim=1) #(subvideos, frames"vectors") -> (subvideos, 1) index should be between 0 and 24
        keyVectors = x_encoder[range(x_encoder.shape[0]),indecies, :] # -> (subvideos,Encoder_out_features)

        r = self.normKeyFrame(keyVectors)
        # x_fc = self.relu(x_fc)

        # *************** this is default code********************
        r_plus_g = r+g  # -> (subvideos, Encoder_out_features*2)

        output = self.fc(r_plus_g)  # -> (subvideos, 4)

        return output
class KeyFramePlusRNormedToMatchGMag(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, alpha=1, Encoder_CheckPoint=None):
        super(KeyFramePlusRNormedToMatchGMag, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)

        self.Key = nn.Linear(self.encoder_out_features,1)# the highest key will determine the output vector
        self.normKeyFrame = nn.BatchNorm1d(self.encoder_out_features) # I am expecting to have one vector with size
        # self.relu = nn.ReLU()
        self.alpha = alpha
        # (vectore from sequence + vector from non-sequence) = encoder_out_features+25
        self.fc = nn.Linear(self.encoder_out_features, num_classes)


    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x_encoder = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x_encoder.shape
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        g = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        #(subvideos, frames"vectors", Encoder_out_features) ->
        # (subvideos*frames"vectors", Encoder_out_features)
        x = x_encoder.view((-1,x_shape[-1]))

        #(subvideos * frames"vectors", Encoder_out_features) -> (subvideos*frames"vectors", 1)
        x_keys = self.Key(x).squeeze().view((x_shape[0],x_shape[1]))# ->(subvideos*frames"vectors") -> (subvideos, frames"vectors")
        _,indecies = x_keys.max(dim=1) #(subvideos, frames"vectors") -> (subvideos, 1) index should be between 0 and 24
        keyVectors = x_encoder[range(x_encoder.shape[0]),indecies, :] # -> (subvideos,Encoder_out_features)

        r = self.normKeyFrame(keyVectors)
        # x_fc = self.relu(x_fc)

        # *************** this is default code********************
        r_normed = torch.linalg.vector_norm(g,dim=1,keepdim=True) * (r/torch.linalg.vector_norm(r,dim=1,keepdim=True))
        r_plus_g = r_normed+g  # -> (subvideos, Encoder_out_features)

        output = self.fc(r_plus_g)  # -> (subvideos, 4)

        return output
class ResNet_subVideo_KeyFramePlusRNormed(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, alpha=1, Encoder_CheckPoint=None):
        super(ResNet_subVideo_KeyFramePlusRNormed, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)

        self.Key = nn.Linear(self.encoder_out_features,1)# the highest key will determine the output vector
        self.normKeyFrame = nn.BatchNorm1d(self.encoder_out_features) # I am expecting to have one vector with size
        # self.relu = nn.ReLU()
        self.alpha = alpha
        # (vectore from sequence + vector from non-sequence) = encoder_out_features+25
        self.fc = nn.Linear(self.encoder_out_features, num_classes)


    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x_encoder = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x_encoder.shape
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        g = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        #(subvideos, frames"vectors", Encoder_out_features) ->
        # (subvideos*frames"vectors", Encoder_out_features)
        x = x_encoder.view((-1,x_shape[-1]))

        #(subvideos * frames"vectors", Encoder_out_features) -> (subvideos*frames"vectors", 1)
        x_keys = self.Key(x).squeeze().view((x_shape[0],x_shape[1]))# ->(subvideos*frames"vectors") -> (subvideos, frames"vectors")
        _,indecies = x_keys.max(dim=1) #(subvideos, frames"vectors") -> (subvideos, 1) index should be between 0 and 24
        keyVectors = x_encoder[range(x_encoder.shape[0]),indecies, :] # -> (subvideos,Encoder_out_features)

        r = self.normKeyFrame(keyVectors)
        # x_fc = self.relu(x_fc)

        # *************** this is default code********************
        r_normed = self.alpha * (r/torch.linalg.vector_norm(r,dim=1,keepdim=True))
        r_plus_g = r_normed+g  # -> (subvideos, Encoder_out_features)

        output = self.fc(r_plus_g)  # -> (subvideos, 4)

        return output
class ResNet_subVideo_KeyFramePlusAllNormed(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, alpha=1, Encoder_CheckPoint=None):
        super(ResNet_subVideo_KeyFramePlusAllNormed, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)

        self.Key = nn.Linear(self.encoder_out_features,1)# the highest key will determine the output vector
        self.normKeyFrame = nn.BatchNorm1d(self.encoder_out_features) # I am expecting to have one vector with size
        # self.relu = nn.ReLU()
        self.alpha = alpha
        # (vectore from sequence + vector from non-sequence) = encoder_out_features+25
        self.fc = nn.Linear(self.encoder_out_features, num_classes)


    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x_encoder = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x_encoder.shape
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        g = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        #(subvideos, frames"vectors", Encoder_out_features) ->
        # (subvideos*frames"vectors", Encoder_out_features)
        x = x_encoder.view((-1,x_shape[-1]))

        #(subvideos * frames"vectors", Encoder_out_features) -> (subvideos*frames"vectors", 1)
        x_keys = self.Key(x).squeeze().view((x_shape[0],x_shape[1]))# ->(subvideos*frames"vectors") -> (subvideos, frames"vectors")
        _,indecies = x_keys.max(dim=1) #(subvideos, frames"vectors") -> (subvideos, 1) index should be between 0 and 24
        keyVectors = x_encoder[range(x_encoder.shape[0]),indecies, :] # -> (subvideos,Encoder_out_features)

        r = self.normKeyFrame(keyVectors)
        # x_fc = self.relu(x_fc)

        # *************** this is default code********************
        r_normed = r/torch.linalg.vector_norm(r,dim=1,keepdim=True)
        g_normed = g/torch.linalg.vector_norm(g,dim=1,keepdim=True)
        r_plus_g = r_normed+g_normed  # -> (subvideos, Encoder_out_features)

        output = self.fc(r_plus_g)  # -> (subvideos, 4)

        return output
################### KEYFRAME ################################
class ResNet_subVideo_KeyFrame(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50='resnet50',
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_KeyFrame, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)

        self.Key = nn.Linear(self.encoder_out_features,1)# the highest key will determine the output vector
        self.normKeyFrame = nn.BatchNorm1d(self.encoder_out_features) # I am expecting to have one vector with size
        # self.relu = nn.ReLU()
        # (vectore from sequence + vector from non-sequence) = encoder_out_features+25
        self.fc = nn.Linear(self.encoder_out_features*2, num_classes)


    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x_encoder = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x_encoder.shape
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        x_gru = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        #(subvideos, frames"vectors", Encoder_out_features) ->
        # (subvideos*frames"vectors", Encoder_out_features)
        x = x_encoder.view((-1,x_shape[-1]))

        #(subvideos * frames"vectors", Encoder_out_features) -> (subvideos*frames"vectors", 1)
        x_keys = self.Key(x).squeeze().view((x_shape[0],x_shape[1]))# ->(subvideos*frames"vectors") -> (subvideos, frames"vectors")
        _,indecies = x_keys.max(dim=1) #(subvideos, frames"vectors") -> (subvideos, 1) index should be between 0 and 24
        keyVectors = x_encoder[range(x_encoder.shape[0]),indecies, :] # -> (subvideos,Encoder_out_features)

        x_fc = self.normKeyFrame(keyVectors)
        # x_fc = self.relu(x_fc)

        # *************** this is default code********************
        x_cat = torch.cat((x_fc, x_gru), dim=1)  # -> (subvideos, Encoder_out_features*2)

        output = self.fc(x_cat)  # -> (subvideos, 4)

        return output

class ResNet_subVideo_KeyFrameOnly(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_KeyFrameOnly, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        # self.normGRU = nn.BatchNorm1d(self.encoder_out_features)

        self.Key = nn.Linear(self.encoder_out_features,1)# the highest key will determine the output vector
        self.normKeyFrame = nn.BatchNorm1d(self.encoder_out_features) # I am expecting to have one vector with size
        # self.relu = nn.ReLU()
        # (vectore from sequence + vector from non-sequence) = encoder_out_features+25
        self.fc = nn.Linear(self.encoder_out_features, num_classes)


    def forward(self, x):
        # *************this is default code*************

        output_dic = self.SubVideo_Encoder(x)
        x_encoder = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_shape = x_encoder.shape
        # x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        # x_gru = self.normGRU(x_gru)

        # ************ your non-sequence code ********************
        #(subvideos, frames"vectors", Encoder_out_features) ->
        # (subvideos*frames"vectors", Encoder_out_features)
        x = x_encoder.view((-1,x_shape[-1]))

        #(subvideos * frames"vectors", Encoder_out_features) -> (subvideos*frames"vectors", 1)
        x_keys = self.Key(x).squeeze().view((x_shape[0],x_shape[1]))# ->(subvideos*frames"vectors") -> (subvideos, frames"vectors")
        _,indecies = x_keys.max(dim=1) #(subvideos, frames"vectors") -> (subvideos, 1) index should be between 0 and 24
        keyVectors = x_encoder[range(x_encoder.shape[0]),indecies, :] # -> (subvideos,Encoder_out_features)

        x_fc = self.normKeyFrame(keyVectors)
        # x_fc = self.relu(x_fc)

        # *************** this is default code********************
        # x_cat = torch.cat((x_fc, x_gru), dim=1)  # -> (subvideos, Encoder_out_features*2)

        output = self.fc(x_fc)  # -> (subvideos, 4)

        return output
##################### FCVERT ################################
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
####################### FCHORIZ ##############################
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

############################ MAX #################################
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
class ResNet_subVideo_Min(nn.Module):#first proposal
    def __init__(self, num_classes=4, pretrained=False, resnet50=True,
                 feature_extract=False, Encoder_CheckPoint=None):
        super(ResNet_subVideo_Min, self).__init__()
        # Our 2D encoder, We can consider 3D encoder instead
        self.SubVideo_Encoder = SubVideo_Encoder(num_classes=num_classes, pretrained=pretrained, resnet50=resnet50,
                                        feature_extract=feature_extract,Encoder_CheckPoint=Encoder_CheckPoint)
        self.encoder_out_features = self.SubVideo_Encoder.Encoder_out_features # probabily 2048
        self.normGRU = nn.BatchNorm1d(self.encoder_out_features)
        self.normMin = nn.BatchNorm1d(self.encoder_out_features)
        self.drop = nn.Dropout(p=0.5)
        # (vectore from sequence + vector from non-sequence) = encoder_out_features*2
        self.fc = nn.Linear(self.encoder_out_features * 2, num_classes)


    def forward(self, x):
        # *************this is default code*************
        x_shape = x.shape
        output_dic = self.SubVideo_Encoder(x)
        x = output_dic["x"]  # x=(subvideos, frames"vectors", Encoder_out_features)
        x_gru = output_dic["x_gru"]  # x_gru = (subvideos, Encoder_out_features)
        x_gru = self.normGRU(x_gru)
        # ************ your non-sequence code ********************
        x_min, _ = x.min(dim=1)  # -> (subvideos, Encoder_out_features)
        x_min = self.normMin(x_min)
        # *************** this is default code********************
        x_cat = torch.cat((x_min, x_gru), dim=1)  # -> (subvideos, 2*Encoder_out_features)
        x_cat_dropout = self.drop(x_cat)
        output = self.fc(x_cat_dropout)  # -> (subvideos, 4)

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

    def __init__(self, num_classes=4, pretrained=True, resnet50='resnet50',
                 feature_extract=True,Encoder_CheckPoint=None):
        super(SubVideo_Encoder, self).__init__()
        ####### This code is used before 26-Jul-22 however due to AIHA reviewers' comments I have to test other Encoder arch
        # if resnet50:
        #     self.original_ResNet = models.resnet50(pretrained=pretrained)
        # else:
        #     self.original_ResNet = models.resnet101(pretrained=pretrained)
        # self.set_parameter_requires_grad(self.original_ResNet,
        #                                  feature_extract=feature_extract)  # if True freeze all the parameters
        #
        # # load the Encoder checkpoint if any
        # if Encoder_CheckPoint:
        #     self.original_ResNet = loadCheckpoint(self.original_ResNet,Encoder_CheckPoint, num_classes)
        self.original_ResNet,_ = helpers.initialize_model(resnet50,num_classes,feature_extract,Encoder_CheckPoint,pretrained)
        print("Transfering Weights for Encoder...")
        print("best validation accuracy", Encoder_CheckPoint['best_val_acc'])
        self.original_ResNet.load_state_dict(Encoder_CheckPoint['best_model_wts'])

        self.layers = list(self.original_ResNet.children())  # seperate the layers
        if resnet50 =='resnet50':
            self.Encoder_out_features = self.layers[-1].in_features
            self.Encoder = nn.Sequential(*self.layers[:-1])  # combine all layers except the last fc layer
        elif resnet50=='vgg':
            self.Encoder_out_features = self.layers[-1][6].in_features
            del self.layers[-1][6]
            self.Encoder = nn.Sequential(*self.layers)  # combine all layers except the last fc layer

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
