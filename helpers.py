from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import timm
import numpy as np
import os
from models import RN, MyResNet, stridedConv, ZhoDenseNet, ResNet50_GRU, Owais, MLP_Mixer
from models.fixed_subVideoLenght_models import *
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, encoder_checkpoint, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "Myresnet50":
        model_ft = MyResNet.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "RN":
        model_ft = RN.ResNet50_RN(pretrained=use_pretrained, num_classes=num_classes)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.original_ResNet50.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "ResNet50_GRU":
        model_ft = ResNet50_GRU.ResNet50_GRU(pretrained=use_pretrained, feature_extract=feature_extract,
                                             num_classes=num_classes, Encoder_CheckPoint=encoder_checkpoint)
        input_size = 224
    elif model_name == "ResNet50_max":
        model_ft = ResNet50_GRU.ResNet50_max(pretrained=use_pretrained, feature_extract=feature_extract,
                                             num_classes=num_classes, Encoder_CheckPoint=encoder_checkpoint)
    elif model_name == "ResNet50_SimplerGRU":
        model_ft = ResNet50_GRU.ResNet50_SimplerGRU(pretrained=use_pretrained, feature_extract=feature_extract,
                                                    num_classes=num_classes, Encoder_CheckPoint=encoder_checkpoint)
        input_size = 224
    elif model_name == "Owais_ResNet18_LSTM":
        model_ft = Owais.Owais_ResNet18_LSTM(pretrained=use_pretrained, feature_extract=feature_extract,
                                             num_classes=num_classes)
        input_size = 224
    elif model_name == "MlpMixer":
        model_ft = MLP_Mixer.MlpMixer(tokens_mlp_dim=None, channels_mlp_dim=None, n_classes=num_classes,
                                      n_blocks=5, pretrained=True, feature_extract=True)
        input_size = 224
    elif model_name == "ResNet101_GRU":
        model_ft = ResNet50_GRU.ResNet50_GRU(pretrained=use_pretrained, resnet50=False, feature_extract=feature_extract)
        input_size = 224
    elif model_name == "ResNet50_h_initialized_GRU":
        model_ft = ResNet50_GRU.ResNet50_h_initialized_GRU(pretrained=use_pretrained, resnet50=True,
                                                           feature_extract=feature_extract)
        input_size = 224
    elif model_name == "ResNet101_h_initialized_GRU":
        model_ft = ResNet50_GRU.ResNet50_h_initialized_GRU(pretrained=use_pretrained, resnet50=False,
                                                           feature_extract=feature_extract)
        input_size = 224
    elif model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
    elif model_name == "stridedConv":
        model_ft = stridedConv.stridedConv(num_classes)
        input_size = (576, 720)
    elif model_name == "stridedConv_GRU":
        model_ft = stridedConv.stridedConv_GRU(num_classes)
        input_size = (576, 720)
    elif model_name == "ZhoDenseNet":
        model_ft = ZhoDenseNet.ZhoDenseNet(num_classes)
        input_size = (256, 256)
    elif model_name == "ZhoDenseNet2":
        model_ft = ZhoDenseNet.ZhoDenseNet2(num_classes)
        input_size = (256, 256)
    elif model_name == "vit_base_patch16_224":
        # import timm
        model_ft = timm.create_model('vit_base_patch16_224', pretrained=use_pretrained, num_classes=num_classes)
        input_size = (224, 224)
    elif model_name == "mixer_b16_224":
        model_ft = timm.create_model('mixer_b16_224', pretrained=use_pretrained, num_classes=num_classes)
        input_size = (224, 224)
    elif model_name == "ResNet50_subVideo_Avg":
        model_ft = ResNet_subVideo_Avg(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                       feature_extract=feature_extract, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224,224)
    elif model_name == "ResNet_subVideo_Max":
        model_ft = ResNet_subVideo_Max(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                       feature_extract=feature_extract, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224,224)
    elif model_name == "ResNet_subVideo_Min":
        model_ft = ResNet_subVideo_Min(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                       feature_extract=feature_extract, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224,224)
    elif model_name == "ResNet_subVideo_GRU":
        model_ft = ResNet_subVideo_GRU(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                       feature_extract=feature_extract, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224, 224)
    elif model_name == "ResNet_subVideo_MaxOnly":
        model_ft = ResNet_subVideo_MaxOnly(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                           feature_extract=feature_extract, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224, 224)
    elif model_name == "ResNet_subVideo_FcHoriz":
        model_ft = ResNet_subVideo_FcHoriz(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                           feature_extract=feature_extract, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224, 224)
    elif model_name == "ResNet_subVideo_FcVert":
        model_ft = ResNet_subVideo_FcVert(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                          feature_extract=feature_extract, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224, 224)
    elif model_name == "ResNet_subVideo_MLP":
        model_ft = ResNet_subVideo_MLP(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                       feature_extract=feature_extract, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224, 224)
    elif model_name == "ResNet_subVideo_MLPOnly":
        model_ft = ResNet_subVideo_MLPOnly(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                           feature_extract=feature_extract, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224, 224)
    elif model_name == "ResNet_subVideo_KeyFrame":
        model_ft = ResNet_subVideo_KeyFrame(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                           feature_extract=feature_extract, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224, 224)
    elif model_name == "ResNet_subVideo_KeyFrameOnly":
        model_ft = ResNet_subVideo_KeyFrameOnly(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                           feature_extract=feature_extract, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224, 224)
    elif model_name == "ResNet_subVideo_KeyFramePlus":
        model_ft = ResNet_subVideo_KeyFramePlus(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                           feature_extract=feature_extract, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224, 224)
    elif model_name == "ResNet_subVideo_KeyFramePlusRNormed":
        model_ft = ResNet_subVideo_KeyFramePlusRNormed(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                           feature_extract=feature_extract, alpha=1, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224, 224)
    elif model_name == "ResNet_subVideo_KeyFramePlusAllNormed":
        model_ft = ResNet_subVideo_KeyFramePlusAllNormed(num_classes=num_classes, pretrained=use_pretrained, resnet50=True,
                                           feature_extract=feature_extract, alpha=1, Encoder_CheckPoint=encoder_checkpoint)
        input_size = (224, 224)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def set_requires_grad_get_optimizer(feature_extract, model_ft, half_freez, print_params=False):
    params_to_update = model_ft.parameters()
    if print_params:
        print("Params to learn:")
    if feature_extract and not half_freez:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(
                    param)  # added only the one that requires grad (all of them are set to false in the function set_parameter_requires_grad())

    elif half_freez:  # make all parameters True, then freez half of them
        params_to_update = []
        for param in model_ft.parameters():
            param.requires_grad = True
            params_to_update.append(param)
        if print_params:
            print("The total number of parameters = {}".format(len(params_to_update)))
            print("param.requires_grad = False for the first half of Network")
        stop = 1
        for i in range(len(params_to_update)):
            params_to_update[i].requires_grad = False
            stop += 1
            if stop > int(len(params_to_update) / 6 * 5):  # if half the wieghts is reached stop
                break
        del params_to_update[0:int(len(params_to_update) / 6 * 5) + 1]
        if print_params:
            print("Total of learnable parameters={}".format(len(params_to_update)))
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            if print_params:
                print("\t", name)

    # Observe that all parameters are being optimized


def get_criterion(device):
    # Setup the loss fxn
    if device == None:
        print("CrossEntropyLoss() is used")
        criterion = nn.CrossEntropyLoss()
    else:
        weights = [0.3072463768, 0.1594202899, 0.2782608696, 0.2550724638]
        class_weights = torch.FloatTensor(weights).to(device)
        # Setup the loss fxn
        print("CrossEntropyLoss(weight=class_weights) is used")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    return criterion


def plot_result(num_epochs, results_dic, model_name, colab_dir):
    # Plot the training curves of validation accuracy & loss
    val_acc = [h.cpu().numpy() for h in results_dic["val_acc_history"]]
    val_loss = [h for h in results_dic["val_loss_history"]]
    train_acc = [h.cpu().numpy() for h in results_dic["train_acc_history"]]
    train_loss = [h for h in results_dic["train_loss_history"]]

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    plt.rc('font', **font)
    plt.title("Accuracy & Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.figure(figsize=(20, 10))
    plt.plot(range(1, num_epochs + 1), val_acc, "b", label="Validation Acc")
    plt.plot(range(1, num_epochs + 1), val_loss, "--b", label="Validation Loss")
    plt.plot(range(1, num_epochs + 1), train_acc, "k", label="Train Acc")
    plt.plot(range(1, num_epochs + 1), train_loss, "--k", label="Train Loss")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 10))
    plt.yticks(np.arange(0, 1.5, 0.1))
    plt.legend()
    plt.grid(True)
    plt.savefig(colab_dir + "/results/" + model_name + ".png")
    plt.clf()
    plt.close('all')

