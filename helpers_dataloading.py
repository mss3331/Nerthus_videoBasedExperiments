import torch
import os
from torchvision import datasets, transforms


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
def get_dataloaders_SubVideoBased():
    # Create Dataset for each video
    #list of subvideos:
    #class 0 = [1_0_0, 1_0_1, 2_0_0, 2_0_1, 2_0_2]
    #class 1 = [3_1_0, 3_1_1, 3_1_2,4_1_0, 4_1_1,5_1_0, 5_1_1, 5_1_2,6_1_0, 6_1_1, 6_1_2,7_1_0, 7_1_1,
    # 8_1_0, 8_1_1, 8_1_2,9_1_0, 9_1_1, 9_1_2, 10_1_0, 10_1_1, 10_1_2,11_1_0, 11_1_1, 12_1_0, 12_1_1]
    #class 2 = [13_2_0, 13_2_1, 14_2_0, 14_2_1, 15_2_0, 15_2_1, 15_2_2, 16_2_0, 16_2_1]
    #class 3 = [17_3_0, 17_3_1, 17_3_2, 18_3_0, 18_3_1, 19_3_0, 19_3_1, 20_3_0, 20_3_1, 20_3_2, 21_3_0, 21_3_1]

    #train

    #val

    # Create Dataloaders

    pass