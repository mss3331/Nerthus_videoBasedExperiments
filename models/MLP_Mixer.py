import einops
import torch
import torch.nn as nn
from torchvision import models


class MlpBlock(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    dim : int
        Input and output dimension of the entire block. Inside of the mixer
        it will either be equal to `n_patches` or `hidden_dim`.

    mlp_dim : int
        Dimension of the hidden layer.

    Attributes
    ----------
    linear_1, linear_2 : nn.Linear
        Linear layers.

    activation : nn.GELU
        Activation.
    """

    def __init__(self, dim, mlp_dim=None):
        super().__init__()

        mlp_dim = dim if mlp_dim is None else mlp_dim
        self.linear_1 = nn.Linear(dim, mlp_dim)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(n_samples, n_channels, n_patches)` or
            `(n_samples, n_patches, n_channels)`.

        Returns
        -------
        torch.Tensor
            Output tensor that has exactly the same shape as the input `x`.
        """
        x = self.linear_1(x)  # (n_samples, *, mlp_dim)
        x = self.activation(x)  # (n_samples, *, mlp_dim)
        x = self.linear_2(x)  # (n_samples, *, dim)
        return x


class MixerBlock(nn.Module):
    """Mixer block that contains two `MlpBlock`s and two `LayerNorm`s.

    Parameters
    ----------
    n_patches : int
        Number of patches the image is split up into.

    hidden_dim : int
        Dimensionality of patch embeddings.

    tokens_mlp_dim : int
        Hidden dimension for the `MlpBlock` when doing token mixing.

    channels_mlp_dim : int
        Hidden dimension for the `MlpBlock` when doing channel mixing.

    Attributes
    ----------
    norm_1, norm_2 : nn.LayerNorm
        Layer normalization.

    token_mlp_block : MlpBlock
        Token mixing MLP.

    channel_mlp_block : MlpBlock
        Channel mixing MLP.
    """

    def __init__(
        self, *, n_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim
    ):
        super().__init__()

        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)

        self.token_mlp_block = MlpBlock(n_patches, tokens_mlp_dim)
        self.channel_mlp_block = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_patches, hidden_dim)`.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as `x`, i.e.
            `(n_samples, n_patches, hidden_dim)`.
        """
        y = self.norm_1(x)  # (n_samples, n_patches, hidden_dim)
        y = y.permute(0, 2, 1)  # (n_samples, hidden_dim, n_patches)
        y = self.token_mlp_block(y)  # (n_samples, hidden_dim, n_patches)
        y = y.permute(0, 2, 1)  # (n_samples, n_patches, hidden_dim)
        x = x + y  # (n_samples, n_patches, hidden_dim)
        y = self.norm_2(x)  # (n_samples, n_patches, hidden_dim)
        res = x + self.channel_mlp_block(y)  # (n_samples, n_patches, hidden_dim)
        return res


class MlpMixer(nn.Module):
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
    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def __init__(self, *, tokens_mlp_dim, channels_mlp_dim, n_classes,
                 n_blocks, pretrained = True, feature_extract = True):
        # number of patches is equivalant to number of images if applied for a video
        super().__init__()
        # n_patches = (image_size // patch_size) ** 2 # assumes divisibility
        self.n_patches = 125 # this is the maximium number of images for a subvideo
        hidden_dim = 512 #number of features that ResNet18 provide
        # No need for embeding since we have an Encoder (ResNet18)
        # self.patch_embedder = nn.Conv2d(
        #     3,
        #     hidden_dim,
        #     kernel_size=patch_size,
        #     stride=patch_size,
        # )
        self.original_ResNet = models.resnet18(pretrained=pretrained)
        # if True freeze all the parameters
        self.set_parameter_requires_grad(self.original_ResNet, feature_extract=feature_extract)
        self.layers = list(self.original_ResNet.children())  # seperate the layers
        self.Encoder = nn.Sequential(*self.layers[:-1])  # combine all layers

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
        self.head_classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, labels, subvideo_lengths):
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
        x = self.Encoder(x)
        x = x.squeeze() #->(N,512)
        #now we want to split the series of frames into subvideos of size 125 images each. a subvideo with a smaller than
        # 125 images should be paaded
        all_subvideos = []
        # subvideo_sequences = split_seq_frames(x, labels)
        subvideo_sequences=torch.split(x, subvideo_lengths, dim=0)
        for subvideo in subvideo_sequences: # for each subvideo frames do
            missing_vectors_num = 0 #the default is that the subvideo contains 125 frames
            if len(subvideo) <= self.n_patches:#if the number of frames is less than expected, padd it
                missing_vectors_num = self.n_patches - len(subvideo)
            else:
                print("*"*50,"Error, subvideo frames are bigger than the expected=",
                        self.n_patches," while recieved=", len(subvideo))
                exit(-1)
            padding = torch.zeros((missing_vectors_num,subvideo.shape[-1])).to(torch.device("cuda:0"))
            subvideo = torch.cat((subvideo,padding))
            all_subvideos.append(subvideo)
        x = torch.stack(all_subvideos) # => (n, images=125, c=512)


        for mixer_block in self.blocks:
            x = mixer_block(x)  # (n_samples, n_patches, hidden_dim) is equivlant to (n_subvideos, n_frames, hidden_dim)

        x = self.pre_head_norm(x)  # (n_samples, n_patches, hidden_dim)
        x = x.mean(dim=1)  # (n_samples, hidden_dim) global average pooling in the original paper
        y = self.head_classifier(x)  # (n_samples, n_classes)
        y_expanded = []
        for i,subvideo in enumerate(subvideo_sequences):  # for each subvideo frames do
            y_expanded.append(y[i].expand(len(subvideo),-1))
        y = torch.cat(y_expanded)
        # print(y.shape)
        return y


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
