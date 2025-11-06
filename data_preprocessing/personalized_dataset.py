import copy

import numpy as np
import torch.utils.data as data
import torch
from PIL import Image

from data_preprocessing.randaugment import RandAugment



class Dataset_Normal(data.Dataset):

    def __init__(self, data, targets, dataset, transform=None, target_transform=None):

        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]
        if self.dataset in ['SVHN']:
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))    
        elif self.dataset in ['fmnist']:
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


class Dataset_with_SampleIndex(data.Dataset):

    def __init__(self, data, targets, dataset, transform=None, target_transform=None):

        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]
        if self.dataset in ['SVHN']:
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))    
        elif self.dataset in ['fmnist']:
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets, index

    def __len__(self):
        return len(self.data)



class Dataset_WeakStrong(data.Dataset):

    def __init__(self, data, targets, ulb, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.ulb = ulb
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

        self.strong_transform = copy.deepcopy(transform)
        self.strong_transform.transforms.insert(0, RandAugment(3, 5, self.dataset))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, target = self.data[index], self.targets[index]
        if self.dataset in ['SVHN']:
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))    
        elif self.dataset in ['fmnist']:
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            sample_transformed = self.transform(img)
            strong_sample_transformed = self.strong_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return ( sample_transformed, strong_sample_transformed, target) if not self.ulb else (
            sample_transformed, strong_sample_transformed, target)
    def __len__(self):
        return len(self.data)

class Dataset_Relabel(data.Dataset):

    def __init__(self, data, targets, ulb, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.ulb = ulb
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

        self.strong_transform = copy.deepcopy(transform)
        self.strong_transform.transforms.insert(0, RandAugment(3, 5, self.dataset))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, target = self.data[index], self.targets[index]
        if self.dataset in ['SVHN']:
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))    
        elif self.dataset in ['fmnist']:
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            sample_transformed = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, sample_transformed, target)
    def __len__(self):
        return len(self.data)
    
    
    

class Dataset_WeakStrong_Redundant(data.Dataset):

    def __init__(self,  data, targets, ulb, dataset, supplement_num=None, transform=None, target_transform=None):
        self.dataset = dataset
        self.ulb = ulb
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.supplement_num = supplement_num
        self._supplement()

        self.strong_transform = copy.deepcopy(transform)
        self.strong_transform.transforms.insert(0, RandAugment(3, 5, self.dataset))
    def _supplement(self):
        if self.supplement_num is not None:
            indics = np.random.choice(len(self.data), self.supplement_num)
            self.data = np.concatenate([self.data, self.data[indics]], axis=0)
            self.targets = np.concatenate([self.targets, self.targets[indics]], axis=0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, target = self.data[index], self.targets[index]
        if self.dataset in ['SVHN']:
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))    
        elif self.dataset in ['fmnist']:
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            sample_transformed = self.transform(img)
            strong_sample_transformed = self.strong_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return ( sample_transformed, strong_sample_transformed, target) if not self.ulb else (
            sample_transformed, strong_sample_transformed)
    def __len__(self):
        return len(self.data)
    

class Constructed_Dataset(data.Dataset):

    def __init__(self, data, aug_data, targets, transform=None, target_transform=None):

        self.data = data
        self.aug_data = data
        self.targets = targets


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, img_aug, targets = torch.Tensor(self.data[index]), torch.Tensor(self.aug_data[index]),\
                                torch.Tensor(self.targets[index])

        return img, img_aug, targets

    def __len__(self):
        return len(self.data)
    

class Imp_Local_Dataset(data.Dataset):

    def __init__(self, data, targets, true_targets, args, transform=None, target_transform=None):

        self.data = data
        self.transform = transform
        self.true_targets = true_targets
        self.targets = targets
        self.args=args



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        
        return img,  target


    def __len__(self):
        return len(self.data)



class Feature_Dataset(data.Dataset):

    def __init__(self, data, targets):

        self.data = data
        self.targets = targets



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        feat, targets = self.data[index], self.targets[index]

        return feat, targets

    def __len__(self):
        return self.data.shape[0]
