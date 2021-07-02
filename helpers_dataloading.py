import torch
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import matplotlib.image as mpimg
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset

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

def createDataSetFromList(data_dir,input_size,folders_name,load_to_RAM):
    '''Recieve lsit of folder names and return a concatinated dataset'''

    dataset_list = []
    for subvideo_name in folders_name:
        # create a dataset based on subvideo_name
        dataset_list.append(Nerthus_SubVideo_Dataset(data_dir + subvideo_name, input_size,load_to_RAM))
    dataset = ConcatDataset(dataset_list)

    return dataset


def _get_all_folders_name():
    class_0 =['/0/1_0_0', '/0/1_0_1', '/0/2_0_0', '/0/2_0_1', '/0/2_0_2']

    class_1 =['/1/3_1_0', '/1/3_1_1', '/1/3_1_2', '/1/4_1_0', '/1/4_1_1',
              '/1/5_1_0', '/1/5_1_1', '/1/5_1_2', '/1/6_1_0', '/1/6_1_1',
              '/1/6_1_2', '/1/7_1_0', '/1/7_1_1', '/1/8_1_0', '/1/8_1_1',
              '/1/8_1_2', '/1/9_1_0', '/1/9_1_1', '/1/9_1_2', '/1/10_1_0',
              '/1/10_1_1', '/1/10_1_2', '/1/11_1_0', '/1/11_1_1', '/1/12_1_0',
              '/1/12_1_1']

    class_2 =['/2/13_2_0', '/2/13_2_1', '/2/14_2_0', '/2/14_2_1', '/2/15_2_0',
              '/2/15_2_1', '/2/15_2_2', '/2/16_2_0', '/2/16_2_1']

    class_3 =['/3/17_3_0', '/3/17_3_1', '/3/17_3_2', '/3/18_3_0', '/3/18_3_1',
              '/3/19_3_0', '/3/19_3_1', '/3/20_3_0', '/3/20_3_1', '/3/20_3_2',
              '/3/21_3_0', '/3/21_3_1']
    return class_0+class_1+class_2+class_3

def howToSplitSubVideos (train_folders, val_folders, shuffle_entire_subvideos, data_dir, input_size, load_to_RAM):
    folders = _get_all_folders_name()
    videos_len = len(folders)
    print("number of subvideos involved the experiment =", videos_len)

    # shuffle the entire dataset into train\val frame 0.8 means 80% for training
    if shuffle_entire_subvideos.find("Frame") == 0:
        '''This function split train test randomely'''
        print("dataset is splitted randomely")
        TTR = float(shuffle_entire_subvideos.split(" ")[-1])
        dataset = createDataSetFromList(data_dir, input_size, folders, load_to_RAM)
        dataset_size = len(dataset)
        np.random.seed(0)
        dataset_permutation = np.random.permutation(dataset_size)
        np.random.seed(0)
        train_dataset = torch.utils.data.Subset(dataset, dataset_permutation[:int(TTR * dataset_size)])
        val_dataset = torch.utils.data.Subset(dataset, dataset_permutation[int(TTR * dataset_size):])
        print("training indices {}\n val indices {}".format(train_dataset.indices[:5], val_dataset.indices[:5]))
        return train_dataset, val_dataset

    # if true, don't consider this split, concat train\val folders, and shuffle the subvideos
    if shuffle_entire_subvideos == "True":
        np.random.seed(0)
        np.random.shuffle(folders)
        np.random.seed(0)
        train_folders = folders[:videos_len // 2]  # 50% for train and the rest of val
        val_folders = folders[videos_len // 2:]
    elif shuffle_entire_subvideos == "Equal":
        # print("video1_0 for train and video1_1 for val, we expect 100% val accuracy")
        train_folders = folders[::2]  # 50% for train and the rest of val
        val_folders = folders[1::2]

    train_dataset = createDataSetFromList(data_dir, input_size, train_folders, load_to_RAM)
    val_dataset = createDataSetFromList(data_dir, input_size, val_folders, load_to_RAM)

    return train_dataset, val_dataset


def get_dataloaders_SubVideoBased(input_size,batch_size,data_dir, load_to_RAM, shuffle=False,shuffle_entire_subvideos=False):
    # Create Dataset for each video
    '''list of subvideos:
    #class 0 = [1_0_0, 1_0_1, 2_0_0, 2_0_1, 2_0_2]
     class 1 = [3_1_0, 3_1_1, 3_1_2,4_1_0, 4_1_1,5_1_0, 5_1_1, 5_1_2,6_1_0, 6_1_1, 6_1_2,7_1_0, 7_1_1,
                8_1_0, 8_1_1, 8_1_2,9_1_0, 9_1_1, 9_1_2, 10_1_0, 10_1_1, 10_1_2,11_1_0, 11_1_1, 12_1_0, 12_1_1]
     class 2 = [13_2_0, 13_2_1, 14_2_0, 14_2_1, 15_2_0, 15_2_1, 15_2_2, 16_2_0, 16_2_1]
     class 3 = [17_3_0, 17_3_1, 17_3_2, 18_3_0, 18_3_1, 19_3_0, 19_3_1, 20_3_0, 20_3_1, 20_3_2, 21_3_0, 21_3_1]'''


    # data_dirEntery_list = list[os.scandir(data_dir)]
    #train
    train_folders = ["/0/2_0_0", "/0/2_0_1", "/0/2_0_2", #class 0
               "/1/3_1_0", "/1/3_1_1", "/1/3_1_2", #class 1
               "/2/15_2_0", "/2/15_2_1", "/2/15_2_2", #class 2
               "/3/17_3_0", "/3/17_3_1", "/3/17_3_2"] #class 3
    # val
    val_folders = ["/0/1_0_0", "/0/1_0_1",  # class 0
                   "/1/4_1_0", "/1/4_1_1",  # class 1
                   "/2/13_2_0", "/2/13_2_1",  # class 2
                   "/3/18_3_0", "/3/18_3_1", ]  # class 3

    # if true, don't consider this split, concat train\val folders, and shuffle the subvideos
    if shuffle_entire_subvideos != None:
        train_dataset, val_dataset = howToSplitSubVideos(train_folders, val_folders,
                                                         shuffle_entire_subvideos,
                                                         data_dir, input_size,
                                                         load_to_RAM)

    print("Training images:", len(train_dataset))
    print("Val images:", len(val_dataset))
    # show_random_samples(train_dataset, 0)
    # show_random_samples(val_dataset, 0)
    # # show_random_samples(train_dataset,0)
    # exit(0)
    image_datasets = {'train':train_dataset, 'val':val_dataset}
    # Create Dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=2)
        for x in ['train', 'val']}
    # show_random_samples(list(dataloaders_dict['train'])[0],0)
    return dataloaders_dict


class Nerthus_SubVideo_Dataset(Dataset):
    def __init__(self, imageDir,targetSize,load_to_RAM= False):

        self.imageList = glob.glob(imageDir +'/*.jpg')
        self.imageList.sort()
        # self.labels = np.arange(len(self.imageList))
        # print((self.imageList[0].split("score_")[1].split("-")[0]))
        self.target_labels = torch.empty(len(self.imageList), dtype=torch.long)
        # self.labels[:] = np.long((self.imageList[0].split("_")[-2].split("-")[0])) # C:\...\0\bowel_20_score_3-1_00000001
        # self.target_labels = torch.from_numpy(self.labels)
        self.target_labels[:] = int(self.imageList[0].split("_")[-2].split("-")[0])
        self.targetSize = targetSize
        self.tensor_images = []

        self.load_to_RAM = load_to_RAM
        if self.load_to_RAM:# load all data to RAM for faster fetching
            print("Loading dataset to RAM...")
            self.tensor_images = [self.get_tensor_image(image_path) for image_path in self.imageList]
            # print("Finish loading dataset to RAM")

    def __getitem__(self, index):
        if self.load_to_RAM:#if images are loaded to the RAM copy them, otherwise, read them
            x = self.tensor_images[index]
        else:
            x=self.get_tensor_image(self.imageList[index])

        return x, self.target_labels[index], self.imageList[index]

    def __len__(self):
        return len(self.imageList)

    def get_tensor_image(self, image_path):
        '''this function get image path and return transformed tensor image'''
        preprocess = transforms.Compose([
            # transforms.Resize((384, 288), 2),
            transforms.Resize(self.targetSize),
            transforms.CenterCrop(self.targetSize),
            transforms.ToTensor()])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        X = Image.open(image_path).convert('RGB')
        X = preprocess(X)
        return X

def show_random_samples(training_data,offset):
    figure = plt.figure(figsize=(16, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        # sample_idx = torch.randint(len(training_data), size=(1,)).item()
        sample_idx = i-1+offset
        if sample_idx == 125:
            print(sample_idx)
        img, label, file_path = training_data[sample_idx]

        figure.add_subplot(rows, cols, i)
        name = file_path.split("\\")[-1].split("_")[-1]
        plt.title(str(label.item())+":"+name)
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0) )
    plt.show()