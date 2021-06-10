import helpers_dataloading
from torch.utils.data import ConcatDataset, RandomSampler, DataLoader

if __name__== '__main__':
    data_dir = r"E:\Databases\Nerthus\SubVideoBased_not_splitted_into_trainVal"
    dataloader_dic = helpers_dataloading.get_dataloaders_SubVideoBased((255,255),9,data_dir,shuffle=False)
    images, labels, filenames = list(dataloader_dic['train'])[0]
    print(len(list(dataloader_dic['train'])))
    temp = zip(images, labels, filenames)
    helpers_dataloading.show_random_samples(list(temp), 0)

    dataloader_dic['train'] =  DataLoader(dataloader_dic['train'].dataset,shuffle=True,
                                          batch_size=dataloader_dic['train'].batch_size)
    images, labels, filenames = list(dataloader_dic['train'])[0]
    print(len(list(dataloader_dic['train'])))
    temp = zip(images, labels, filenames)
    helpers_dataloading.show_random_samples(list(temp), 0)

