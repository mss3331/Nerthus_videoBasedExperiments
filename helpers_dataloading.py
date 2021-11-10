import torch
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
from torch.utils.data import ConcatDataset
import matplotlib.image as mpimg
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset


def my_collate(data):
    # data is a list of batches each batch contains imgs,labels,file_names
    imgs = []
    labels = []
    file_names = []
    subvideo_lengths = []
    for subvideo in data:
        imgs += subvideo[0]
        labels += subvideo[1]
        file_names += subvideo[2]
        subvideo_lengths.append(len(subvideo[1]))
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    return [imgs, labels, file_names, subvideo_lengths]


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


def get_dataloaders(input_size, batch_size, data_dir):
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


def get_dataloaders_Kvasir(input_size, batch_size, data_dir, shuffle):
    TTR = 0.5
    data_transforms = get_transform_conf(input_size=input_size)
    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    kvasir_dataset = datasets.ImageFolder(data_dir, data_transforms["train"])

    dataset_size = len(kvasir_dataset)
    np.random.seed(0)
    dataset_permutation = np.random.permutation(dataset_size)
    np.random.seed(0)
    train_dataset = torch.utils.data.Subset(kvasir_dataset, dataset_permutation[:int(TTR * dataset_size)])
    val_dataset = torch.utils.data.Subset(kvasir_dataset, dataset_permutation[int(TTR * dataset_size):])
    print("Training images:", len(train_dataset))
    print("Val images:", len(val_dataset))
    print("training indices {}\n val indices {}".format(train_dataset.indices[:5], val_dataset.indices[:5]))
    train_val_dataset = {"train": train_dataset, "val": val_dataset}

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(train_val_dataset[x], batch_size=batch_size, shuffle=shuffle, num_workers=2)
        for x in ['train', 'val']}
    return dataloaders_dict


def createDataSetFromList(data_dir, input_size, folders_name, load_to_RAM, EntireSubVideo, sub_videoSize):
    '''Recieve lsit of folder names and return a concatinated dataset'''

    dataset_list = []
    for subvideo_name in folders_name:
        # create a dataset based on subvideo_name
        if EntireSubVideo == "FixedSize":
            dataset_list.append(Nerthus_EntireSubVideo_FromImageBased_Dataset(data_dir + subvideo_name, input_size, sub_videoSize, load_to_RAM))
        elif EntireSubVideo:
            dataset_list.append(Nerthus_EntireSubVideo_Dataset(data_dir + subvideo_name, input_size, load_to_RAM))
        else:
            dataset_list.append(Nerthus_SubVideo_Dataset(data_dir + subvideo_name, input_size, load_to_RAM))
    dataset = ConcatDataset(dataset_list)

    return dataset


def _get_all_folders_name(data_dir, shuffle_entire_subvideos):
    # class_0 =['/0/1_0_0', '/0/1_0_1', '/0/2_0_0', '/0/2_0_1', '/0/2_0_2']
    #
    # class_1 =['/1/3_1_0', '/1/3_1_1', '/1/3_1_2', '/1/4_1_0', '/1/4_1_1',
    #           '/1/5_1_0', '/1/5_1_1', '/1/5_1_2', '/1/6_1_0', '/1/6_1_1',
    #           '/1/6_1_2', '/1/7_1_0', '/1/7_1_1', '/1/8_1_0', '/1/8_1_1',
    #           '/1/8_1_2', '/1/9_1_0', '/1/9_1_1', '/1/9_1_2', '/1/10_1_0',
    #           '/1/10_1_1', '/1/10_1_2', '/1/11_1_0', '/1/11_1_1', '/1/12_1_0',
    #           '/1/12_1_1']
    #
    # class_2 =['/2/13_2_0', '/2/13_2_1', '/2/14_2_0', '/2/14_2_1', '/2/15_2_0',
    #           '/2/15_2_1', '/2/15_2_2', '/2/16_2_0', '/2/16_2_1']
    #
    # class_3 =['/3/17_3_0', '/3/17_3_1', '/3/17_3_2', '/3/18_3_0', '/3/18_3_1',
    #           '/3/19_3_0', '/3/19_3_1', '/3/20_3_0', '/3/20_3_1', '/3/20_3_2',
    #           '/3/21_3_0', '/3/21_3_1']
    # if is_subsub_videos:#include sub sub videos such as 1_0_1_3
    #     class_0 +=[]
    # return (class_0,class_1,class_2,class_3)

    all_classes_dir = sorted(glob.glob(data_dir + "/*/"))  # list all stools folders [0, 1, 2, 3]
    print(all_classes_dir)
    folder_list = []
    for class_dir in all_classes_dir:
        if shuffle_entire_subvideos == "TrueWithinClass":
            temp_list = glob.glob(class_dir + "/*/")
            temp_list.sort()
            np.random.seed(0)
            np.random.shuffle(temp_list)
            np.random.seed(0)
        else:
            temp_list = sorted(glob.glob(class_dir + "/*/"),
                           key=lambda x: int(x.split('/')[-2].split('_')[0])*10+int(x.split('/')[-2].split('_')[-1]))  # list all sub videos name

        # in Colab the path is ./content/Nerthus so convert it to \\ .\\content\\ like windows
        temp_list = ["\\".join(folder.split('/')) for folder in temp_list]
        temp_list = ["/" + "/".join(folder.split("\\")[-3:]) for folder in temp_list]
        folder_list += temp_list

    print(folder_list)
    return folder_list


def howToSplitSubVideos(train_folders, val_folders, shuffle_entire_subvideos, data_dir, input_size,
                        load_to_RAM, EntireSubVideo, is_subsub_videos, sub_videoSize):
    folders = _get_all_folders_name(data_dir, shuffle_entire_subvideos)
    # folders_combined = np.concatenate(folders)
    folders_combined = folders  # the folders are sorted from _get_all_folders_name function
    videos_len = len(folders_combined)
    print("number of subvideos involved the experiment =", videos_len)

    # shuffle the entire dataset into train\val frame 0.8 means 80% for training
    if shuffle_entire_subvideos.find("Frame") == 0:
        '''This function split train test randomely'''
        print("dataset is splitted randomely")
        TTR = float(shuffle_entire_subvideos.split(" ")[-1])  # Frame 0.8 --> TTR=0.8
        dataset = createDataSetFromList(data_dir, input_size, folders_combined, load_to_RAM, EntireSubVideo)
        dataset_size = len(dataset)
        np.random.seed(0)
        dataset_permutation = np.random.permutation(dataset_size)
        np.random.seed(0)
        train_dataset = torch.utils.data.Subset(dataset, dataset_permutation[:int(TTR * dataset_size)])
        val_dataset = torch.utils.data.Subset(dataset, dataset_permutation[int(TTR * dataset_size):])
        print("training indices {}\n val indices {}".format(train_dataset.indices[:5], val_dataset.indices[:5]))
        return train_dataset, val_dataset

    # if true, don't consider this split, concat train\val folders, and shuffle the subvideos
    if shuffle_entire_subvideos == "True":  # this is shuffling the Entire subvideos, we may get 100% val if we are lucky (i.e. video 1_0_1 train while 1_0_0 val)
        np.random.seed(0)
        np.random.shuffle(folders_combined)
        np.random.seed(0)
        train_folders = folders_combined[:videos_len // 2]  # 50% for train and the rest of val
        val_folders = folders_combined[videos_len // 2:]
    elif shuffle_entire_subvideos == "TrueWithinClass":  # if a class has video1_0,video1_1,video2_0,video2_1 it will be shuffled and divided between train/val
        train_folders = folders_combined[::2]  # 50% for train and the rest of val
        val_folders = folders_combined[1::2]
    elif shuffle_entire_subvideos == "Equal":  # Equal to original Nerthus splitting
        # print("video1_0 for train and video1_1 for val, we expect 100% val accuracy")
        train_folders = []
        val_folders = []
        # for class_folder in folders: #folders = (class_0, class_1, class_2, class_3)
        #     train_folders+=class_folder[::2]  # 50% for train and the rest of val
        #     val_folders+=class_folder[1::2]
        train_folders = folders_combined[::2]  # 50% for train and the rest of val
        val_folders = folders_combined[1::2]
        # the following is pointless since we are going to shuffle them using the dataloaders anyway
    elif shuffle_entire_subvideos == "None Shuffle":  # consider the current training subvide but only shuffle them.
        np.random.shuffle(train_folders)

    print(train_folders)
    print(val_folders)

    train_dataset = createDataSetFromList(data_dir, input_size, train_folders, load_to_RAM, EntireSubVideo, sub_videoSize)
    val_dataset = createDataSetFromList(data_dir, input_size, val_folders, load_to_RAM, EntireSubVideo, sub_videoSize)

    return train_dataset, val_dataset


def get_base_dataset_train_val_folders_name(is_subsub_videos):
    # train
    train_folders = ["/0/2_0_0", "/0/2_0_1", "/0/2_0_2",  # class 0
                     "/1/3_1_0", "/1/3_1_1", "/1/3_1_2", "/1/5_1_0", "/1/5_1_1", "/1/5_1_2",  # class 1
                     "/2/14_2_0", "/2/14_2_1", "/2/15_2_0", "/2/15_2_1", "/2/15_2_2",  # class 2
                     "/3/17_3_0", "/3/17_3_1", "/3/17_3_2", "/3/19_3_0", "/3/19_3_1", ]  # class 3
    val_folders = ["/0/1_0_0", "/0/1_0_1",  # class 0
                   "/1/4_1_0", "/1/4_1_1", "/1/6_1_0", "/1/6_1_1", "/1/6_1_2",  # class 1
                   "/2/16_2_0", "/2/16_2_1", "/2/13_2_0", "/2/13_2_1",  # class 2
                   "/3/18_3_0", "/3/18_3_1", "/3/20_3_0", "/3/20_3_1", "/3/20_3_2", ]  # class 3
    if is_subsub_videos:  # if the dataset splitted into further subsub videos then add those folders
        train_folders += ["/0/2_0_0_5", "/0/2_0_1_6",
                          "/1/3_1_0_9", "/1/3_1_1_10", "/1/5_1_0_12", "/1/5_1_1_13",
                          "/2/14_2_0_5", "/2/15_2_0_6", "/2/15_2_1_7", "/2/15_2_2_8",
                          "/3/17_3_0_3", "/3/17_3_1_4", "/3/19_3_0_7", "/3/19_3_1_8"
                          ]
        val_folders += ["/0/1_0_0_3", "/0/1_0_1_4",
                        "/1/4_1_0_11", "/1/6_1_0_14", "/1/6_1_1_15",
                        "/2/16_2_0_9", "/2/13_2_0_3", "/2/13_2_1_4",
                        "/3/18_3_0_5", "/3/18_3_1_6", "/3/20_3_0_9", "/3/20_3_1_10"
                        ]
    return (train_folders, val_folders)


def get_dataloaders_SubVideoBased(input_size, batch_size, data_dir, load_to_RAM, is_subsub_videos,
                                  shuffle=False, shuffle_entire_subvideos=False, EntireSubVideo=True, sub_videoSize=25):
    # Create Dataset for each video
    '''list of subvideos:
    #class 0 = [1_0_0, 1_0_1, 2_0_0, 2_0_1, 2_0_2]
     class 1 = [3_1_0, 3_1_1, 3_1_2,4_1_0, 4_1_1,5_1_0, 5_1_1, 5_1_2,6_1_0, 6_1_1, 6_1_2,7_1_0, 7_1_1,
                8_1_0, 8_1_1, 8_1_2,9_1_0, 9_1_1, 9_1_2, 10_1_0, 10_1_1, 10_1_2,11_1_0, 11_1_1, 12_1_0, 12_1_1]
     class 2 = [13_2_0, 13_2_1, 14_2_0, 14_2_1, 15_2_0, 15_2_1, 15_2_2, 16_2_0, 16_2_1]
     class 3 = [17_3_0, 17_3_1, 17_3_2, 18_3_0, 18_3_1, 19_3_0, 19_3_1, 20_3_0, 20_3_1, 20_3_2, 21_3_0, 21_3_1]'''

    # data_dirEntery_list = list[os.scandir(data_dir)]
    train_folders, val_folders = get_base_dataset_train_val_folders_name(is_subsub_videos)

    # if true, don't consider this split, concat train\val folders, and shuffle the subvideos
    if shuffle_entire_subvideos != None:
        train_dataset, val_dataset = howToSplitSubVideos(train_folders, val_folders,
                                                         shuffle_entire_subvideos,
                                                         data_dir, input_size,
                                                         load_to_RAM, EntireSubVideo, is_subsub_videos, sub_videoSize)

    print("Training images:", len(train_dataset))
    print("Val images:", len(val_dataset))
    # show_random_samples(train_dataset, 0)
    # show_random_samples(val_dataset, 0)
    # # show_random_samples(train_dataset,0)
    # exit(0)
    image_datasets = {'train': train_dataset, 'val': val_dataset}
    collate_fn = None
    if EntireSubVideo == "True": collate_fn = my_collate
    # Create Dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                                       num_workers=2)
        for x in ['train', 'val']}
    # show_random_samples(list(dataloaders_dict['train'])[0],0)
    return dataloaders_dict


class Nerthus_SubVideo_Dataset(Dataset):
    def __init__(self, imageDir, targetSize, load_to_RAM=False):

        self.imageList = glob.glob(imageDir + '/*.jpg')
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
        if self.load_to_RAM:  # load all data to RAM for faster fetching
            print("Loading dataset to RAM...")
            self.tensor_images = [self.get_tensor_image(image_path) for image_path in self.imageList]
            # print("Finish loading dataset to RAM")

    def __getitem__(self, index):
        if self.load_to_RAM:  # if images are loaded to the RAM copy them, otherwise, read them
            x = self.tensor_images[index]
        else:
            x = self.get_tensor_image(self.imageList[index])

        return x, self.target_labels[index], self.imageList[index]

    def __len__(self):
        return len(self.imageList)

    def get_tensor_image(self, image_path):
        '''this function get image path and return transformed tensor image'''
        preprocess = transforms.Compose([
            # transforms.Resize((384, 288), 2),
            transforms.Resize(self.targetSize),
            transforms.CenterCrop(self.targetSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        X = Image.open(image_path).convert('RGB')
        X = preprocess(X)
        return X


class Nerthus_EntireSubVideo_Dataset(Dataset):
    def __init__(self, imageDir, targetSize, load_to_RAM=False):

        self.imageList = glob.glob(imageDir + '/*.jpg')
        self.imageList.sort()
        # self.labels = np.arange(len(self.imageList))
        # print((self.imageList[0].split("score_")[1].split("-")[0]))
        self.target_labels = torch.empty(len(self.imageList), dtype=torch.long)
        # self.labels[:] = np.long((self.imageList[0].split("_")[-2].split("-")[0])) # C:\...\0\bowel_20_score_3-1_00000001
        # self.target_labels = torch.from_numpy(self.labels)
        # print("ok",self.imageList[0].split("_"))
        self.target_labels[:] = int(self.imageList[0].split("_")[-2].split("-")[0])
        self.targetSize = targetSize
        self.tensor_images = []

        self.load_to_RAM = load_to_RAM
        if self.load_to_RAM:  # load all data to RAM for faster fetching
            print("Loading dataset to RAM...")
            self.tensor_images = [self.get_tensor_image(image_path) for image_path in self.imageList]
            # print("Finish loading dataset to RAM")

    def __getitem__(self, index):
        if self.load_to_RAM:  # if images are loaded to the RAM copy them, otherwise, read them
            x = self.tensor_images
        else:
            x = self.get_tensor_image(self.imageList[:])

        return torch.stack(x), self.target_labels, self.imageList

    def __len__(self):
        return 1  # len(self.imageList)


    def get_tensor_image(self, image_path, multiple_paths=False):
        '''this function get image path and return transformed tensor image'''
        preprocess = transforms.Compose([
            # transforms.Resize((384, 288), 2),
            transforms.Resize(self.targetSize),
            transforms.CenterCrop(self.targetSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if multiple_paths:
            X = Image.open(image_path).convert('RGB')
            X = preprocess(X)
        else:
            X = [preprocess(Image.open(image).convert('RGB')) for image in image_path]

        return X

class Nerthus_EntireSubVideo_FromImageBased_Dataset(Dataset):
    def __init__(self, imageDir, targetSize, sub_videoSize=25, load_to_RAM=False):
        '''partitions is an integer that divide a video (e.g. 1_0_0) into subvideos
          each subvideo would have frames from each second.
        '''
        self.imageList = glob.glob(imageDir + '/*.jpg')
        self.imageList.sort()
        self.total_images_size = len(self.imageList)
        # self.labels = np.arange(len(self.imageList))
        # print((self.imageList[0].split("score_")[1].split("-")[0]))
        self.target_labels = torch.empty(len(self.imageList), dtype=torch.long)
        # self.labels[:] = np.long((self.imageList[0].split("_")[-2].split("-")[0])) # C:\...\0\bowel_20_score_3-1_00000001
        # self.target_labels = torch.from_numpy(self.labels)
        # print("ok",self.imageList[0].split("_"))
        self.target_labels[:] = int(self.imageList[0].split("_")[-2].split("-")[0])
        subVideInfo = self.makeSubVideos(self.imageList, self.target_labels, partitions= self.total_images_size//sub_videoSize)
        # subVideo_images_list = [[ima1.jpg,img10.jpg],[img2.jpg,img20.jpg]].
        # subVideo_labels_list=[[label1,label10][label2,label20]]= class number
        self.subVideo_images_list, self.subVideo_labels_list, self.subVideo_path_list = subVideInfo

        self.targetSize = targetSize
        self.tensor_images = []

        self.load_to_RAM = load_to_RAM
        if self.load_to_RAM:  # load all data to RAM for faster fetching
            print("Loading dataset to RAM...")
            self.tensor_images = [self.get_tensor_image(image_path) for image_path in self.imageList]
            sys.exit('loading from RAM not implemented yet for this custom dataset')

            # print("Finish loading dataset to RAM")

    def __getitem__(self, index):
        if self.load_to_RAM:  # if images are loaded to the RAM copy them, otherwise, read them
            x = self.tensor_images
        else:
            x = self.get_tensor_image(self.subVideo_images_list[index])

        return torch.stack(x), self.subVideo_labels_list[index][0], self.subVideo_path_list[index]

    def __len__(self):
        return len(self.subVideo_images_list)  # [[img1,img10][img2,img20]] = len = 2 subvideos

    def get_tensor_image(self, image_path, multiple_paths=False):
        '''this function get image path and return transformed tensor image'''
        preprocess = transforms.Compose([
            # transforms.Resize((384, 288), 2),
            transforms.Resize(self.targetSize),
            transforms.CenterCrop(self.targetSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if multiple_paths:
            X = Image.open(image_path).convert('RGB')
            X = preprocess(X)
        else:
            X = [preprocess(Image.open(image).convert('RGB')) for image in image_path]

        return X

    def makeSubVideos(self,imageList, target_labels, partitions,):
        subvideo_images_list = []
        subvideo_labels_list = []
        subvideo_path_list = []

        for shift in range(partitions):
            subvideo_images_list.append(imageList[shift::partitions])
            subvideo_labels_list.append(target_labels[shift::partitions])
            subvideo_path_list.append(imageList[shift::partitions])
        return subvideo_images_list,subvideo_labels_list, subvideo_path_list


def show_random_samples(training_data, offset):
    figure = plt.figure(figsize=(16, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        # sample_idx = torch.randint(len(training_data), size=(1,)).item()
        sample_idx = i - 1 + offset
        if sample_idx == 125:
            print(sample_idx)
        img, label, file_path = training_data[sample_idx]

        figure.add_subplot(rows, cols, i)
        name = file_path.split("\\")[-1].split("_")[-1]
        plt.title(str(label.item()) + ":" + name)
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()