#9064466c4a4f16db52c1672e03ee3c52060a24e4 token
import torch
import torchvision
import helpers
import train_model
import numpy as np
from torchvision import datasets, models, transforms

def run():
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("device = ",device)

    data_dir = r"C:\Users\Mahmood_Haithami\Downloads\JDownloader\Databases\Nerthus\frameBased\frameBased_randomShuffle2"
    # Models to choose from [resnet18,resnet50, alexnet, vgg, squeezenet, densenet, inception
    # Myresnet50,RN]
    model_name = "resnet50"
    # Number of classes in the dataset
    num_classes = 4
    batch_size = 8
    #batch_size = 16
    # Number of epochs to train for
    num_epochs = 10
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False
    half_freez = False

    # Initialize the model for this run
    model_ft, input_size = helpers.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # print(model_ft)
    # exit(0)
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Print the model we just instantiated
    print("the used model is ",model_name)

    dataloaders_dict = helpers.get_dataloaders(input_size,batch_size,data_dir)

    criterion = helpers.get_criterion()
    optimizer_ft = helpers.set_requires_grad_get_optimizer(feature_extract,model_ft,half_freez)
    # Train and evaluate
    model_ft, results_dic = train_model.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,device=device, num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"))
    helpers.plot_result(num_epochs=num_epochs,results_dic=results_dic)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run()