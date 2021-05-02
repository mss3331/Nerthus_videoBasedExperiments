from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from models import RN, MyResNet, stridedConv
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


def set_parameter_requires_grad(model, feature_extracting):

    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False




def initialize_model(model_name, num_classes, feature_extract,use_pretrained=True):
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
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "RN":
        model_ft = RN.ResNet50_FE(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.original_ResNet50.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
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
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
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
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
    elif model_name == "stridedConv":
        model_ft = stridedConv.stridedConv(num_classes)
        input_size = (576,720)
    elif model_name == "stridedConv_GRU":
        model_ft = stridedConv.stridedConv_GRU(num_classes)
        input_size = (576, 720)
    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size


def get_transform_conf(input_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def set_requires_grad_get_optimizer(feature_extract,model_ft,half_freez):
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract and not half_freez:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param) # added only the one that requires grad (all of them are set to false in the function set_parameter_requires_grad())

    elif half_freez: #make all parameters True, then freez half of them
        params_to_update = []
        for param in model_ft.parameters():
            param.requires_grad = True
            params_to_update.append(param)
        print("The total number of parameters = {}".format(len(params_to_update)))
        print("param.requires_grad = False for the first half of Network")
        stop=1
        for i in range(len(params_to_update)):
            params_to_update[i].requires_grad = False
            stop +=1
            if stop > int(len(params_to_update)/6*5):#if half the wieghts is reached stop
                break
        del params_to_update[0:int(len(params_to_update)/6*5)+1]
        print("Total of learnable parameters={}".format(len(params_to_update)))
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)


    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return optimizer_ft

def get_dataloaders(input_size,batch_size,data_dir):

    data_transforms = get_transform_conf(input_size=input_size)
    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4)
        for x in ['train', 'val']}
    return dataloaders_dict

def get_criterion():
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    return criterion



def plot_result(num_epochs,results_dic):

    # Plot the training curves of validation accuracy & loss
    val_acc = [h.cpu().numpy() for h in results_dic["val_acc_history"]]
    val_loss = [h for h in results_dic["val_loss_history"]]
    train_acc = [h.cpu().numpy() for h in results_dic["train_acc_history"]]
    train_loss = [h for h in results_dic["train_loss_history"]]


    plt.title("Accuracy & Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,num_epochs+1),val_acc,"b",label="Validation Acc")
    plt.plot(range(1,num_epochs+1),val_loss,"--b",label="Validation Loss")
    plt.plot(range(1,num_epochs+1),train_acc,"k",label="Train Acc")
    plt.plot(range(1,num_epochs+1),train_loss,"--k",label="Train Loss")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 1))
    plt.yticks(np.arange(0,1.5, 0.1))
    plt.legend()
    plt.grid(True)
    plt.show()
