#9064466c4a4f16db52c1672e03ee3c52060a24e4 token
import torch
import torchvision
import helpers
import helpers_dataloading
import train_model
import numpy as np
import torch.optim as optim
import wandb
wandb.login(key="38818beaffe50c5403d3f87b288d36d1b38372f8")

def print_hyperparameters():
    print("learning_rate {}\n,num_classes {}\n,batch_size {}\n"
          ",num_epochs {}\n load_to_RAM {}".format(learning_rate,num_classes,batch_size,num_epochs,load_to_RAM))
    print("model_name = ",model_name)
    print("shuffle= ", shuffle)
    print("feature_extract",feature_extract)
    print("pretrained=", use_pretrained)
    print("shuffle_entire_subvideos=",shuffle_entire_subvideos)
    if shuffle_entire_subvideos == "True":
        print("This dataset will be shuffled and distributed to train\\val based on "
              "subvideos as in the Original Nerthus paper")
    elif shuffle_entire_subvideos == "Equal":
        print("video1_0 for train and video1_1 for val, we expect 100% val accuracy")
    print("EntireSubVideo", EntireSubVideo)
    print("Did you consider SubSub Videos? ",is_subsub_videos)
    if data_dir.find("kvasir")>=0:
        print("*"*20,"Kvasir Dataset is used here not Nerthus", "*"*20)
    if checkpoint:
        print("a checkpoint is loaded")

def run():
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("device = ",device)


    # Initialize the model for this run
    model_ft, input_size = helpers.initialize_model(model_name, num_classes, feature_extract,checkpoint, use_pretrained=use_pretrained)
    # print(model_ft)
    # exit(0)
    # Send the model to GPU
    model_ft = model_ft.to(device)
    # for name,param in model_ft.named_parameters():
    #     if param.requires_grad == True:
    #         print("\t",name)
    # print('nothing to print')
    # exit(0)
    # Print the model we just instantiated
    print("the used model is ",model_name)

    # dataloaders_dict = helpers_dataloading.get_dataloaders(input_size,batch_size,data_dir)
    # dataloaders_dict = helpers_dataloading.get_dataloaders_SubVideoBased(input_size,batch_size,data_dir,load_to_RAM, shuffle=shuffle)
    if data_dir.find("kvasir")>=0:
        dataloaders_dict = helpers_dataloading.get_dataloaders_Kvasir(input_size,batch_size,data_dir,shuffle)
    else:
        dataloaders_dict = helpers_dataloading.get_dataloaders_SubVideoBased(input_size,batch_size,data_dir,load_to_RAM,is_subsub_videos
                                                                         , shuffle=shuffle
                                                                         , shuffle_entire_subvideos=shuffle_entire_subvideos
                                                                         , EntireSubVideo=EntireSubVideo)
    # if EntireSubVideo:
    #     trainDataset = dataloaders_dict["train"].dataset
    #     valDataset = dataloaders_dict["val"].dataset
    #     dataloaders_dict = {"train":trainDataset,"val":valDataset}
    # print(len(list(dataloaders_dict["train"])[0]))
    # exit(0)
    criterion = helpers.get_criterion()
    # optimizer_ft = helpers.set_requires_grad_get_optimizer(feature_extract,model_ft,half_freez)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
    # if model_name.find("Zho")>=0:
    #     optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate) #only for Zho
    # elif model_name.find("RN"):
    #     optimizer_ft = optim.RMSProp(model_ft.parameters(), lr=learning_rate, weight_decay=8*10**-8) #only for Zho
    # Train and evaluate
    extra_args = (input_size,batch_size,data_dir, load_to_RAM,is_subsub_videos, shuffle,shuffle_entire_subvideos)

    model_ft, results_dic = train_model.train_model(model_ft, dataloaders_dict, criterion,
                                                    optimizer_ft,device,model_name,colab_dir,
                                                    num_epochs=num_epochs,is_inception=(model_name == "inception"),extra_args=extra_args)





if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch.autograd.set_detect_anomaly(True)
    # data_dir = r"E:\Databases\Nerthus\frameBased\frameBased_randomShuffle2"
    data_dir = "E:/Databases/Nerthus/SubSubVideoBased_not_splitted_into_trainVal"
    # data_dir = r"E:\Databases\kvasir-dataset-v2"
    # Colab
    colab_dir = "."
    run_in_colab = True
    # run_in_colab = False
    if run_in_colab:
        data_dir = "/content/SubSubVideoBased_not_splitted_into_trainVal"
        # data_dir = "/content/kvasir-dataset-v2/"
        colab_dir = "/content/Nerthus_videoBasedExperiments/" # base folder for where to store the results
    # data_dir = "/content/frameBased_randomShuffle1"
    # Models to choose from [resnet18,resnet50, alexnet, vgg, squeezenet, densenet, inception
    # Myresnet50,RN,stridedConv,ZhoDenseNet, ResNet50_GRU, ResNet101_GRU, ResNet50_h_initialized_GRU,
    # Owais_ResNet18_LSTM,MlpMixer, ResNet50_max, ResNet50_SimplerGRU]
    model_name = "resnet50"
    # Number of classes in the dataset
    learning_rate = 0.001
    num_classes = 4
    if data_dir.find("kvasir")>=0:
        num_classes = 8
    batch_size = 4
    # batch_size = 2
    # batch_size = 150
    # batch_size = 32 #only for Zho
    num_epochs = 50
    # num_epochs = 1
    load_to_RAM = True
    load_to_RAM = False
    shuffle = True
    #shuffle_entire_subvideos = either None, True, Equal (video1_0 for train and video1_1 for val), Frame 0.5 means 50% for train and 50% for val.
    # "None Shuffle" means consider the base dataset but shuffle the training folder
    #shuffle_entire_subvideos = [None, True, Equal, Frame 0.5, None Shuffle]
    is_subsub_videos = False
    shuffle_entire_subvideos = "Frame 0.8" # if true, the train and val would have shuffeled videos as in the Original Nerthus paper
    # shuffle_entire_subvideos = "True"
    # if EntireSubVideo= true, load entire subvideo as one instance.
    # and shuffle =True means that shuffle the subvideos rather than shuffle the frames in the subvideo
    # and batch_size = 3 means load 3 subvideos
    EntireSubVideo = False
    feature_extract = False
    half_freez = False
    use_pretrained = False
    checkpoint =torch.load(colab_dir+"/checkpoints/resnet50.pth")
    print_hyperparameters()
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params

    wandb.init(
        project="Nerthus",
        entity="mss3331",
        name=ExperimentNumber,
        # Track hyperparameters and run metadata
        config={

            "learning_rate": learning_rate,
            "shuffle": shuffle,
            "is_subsub_videos":is_subsub_videos,
            "shuffle_entire_subvideos":shuffle_entire_subvideos,
            "architecture": model_name,
            "batch_size":batch_size,
            "use_pretrained":use_pretrained,
            "feature_extract":feature_extract,
            "num_epochs":num_epochs,
            "dataset": data_dir.split("/")[-1], })

    run()

