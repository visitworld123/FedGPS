
import torchvision.transforms as transforms


data_stats = {'FMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}

def get_stransform(dataset, 
                   train,
                   normalize=True):

    color_jitter = transforms.ColorJitter(
        0.8 , 0.8, 0.8, 0.2
    )
    if train:
        if dataset in ['FMNIST']:
            transform = transforms.Compose([
                transforms.Resize(32),
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1])])
        elif dataset in ['SVHN']:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1])])
        elif dataset in ['FLAIR']:
            transform = transforms.Compose([
                transforms.Resize(128),
                transforms.RandomCrop(128, padding=10),
                transforms.RandomHorizontalFlip(p=0.9),
                transforms.ToTensor()])
        else:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1])])
    else:
        if dataset in ['FLAIR']:
            transform = transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor()])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(data_stats[dataset][0], data_stats[dataset][1])])
    return transform