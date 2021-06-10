import helpers_dataloading
from torch.utils.data import ConcatDataset

if __name__== '__main__':
    data_dir = r"E:\Databases\Nerthus\SubVideoBased_not_splitted_into_trainVal"
    dataloader_dic = helpers_dataloading.get_dataloaders_SubVideoBased((255,255),9,data_dir,shuffle=True)

    images,labels, filenames = list(dataloader_dic['train'])[102]
    temp = zip(images,labels,filenames)
    helpers_dataloading.show_random_samples(list(temp),0)

