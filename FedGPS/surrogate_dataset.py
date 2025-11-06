import os
import logging
import random
from PIL import Image
# import accimage
import numpy as np
import torch
import torch.utils.data as data
import torchvision


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

import torchvision.transforms as transforms
def surrogate_load_dataset(args):

    batch_size = args.algorithms.surrogate_dataset_batch_size // len(args.algorithms.surrogate_dataset_list)
    image_size = args.algorithms.dataset_image_size

    logging.info(f"Loading datasets for client ------  surrogate_dataset_list :{args.algorithms.surrogate_dataset_list}")

    dataset_list = []
    datadir_list = []

    # May we support multiple dataset in the future.
    for i, dataset_name in enumerate(args.algorithms.surrogate_dataset_list):
        if "c" in dataset_name or "Gaussian" in dataset_name or "decoder" in dataset_name:
            dataset_list.append(dataset_name)
            datadir = os.path.join(args.algorithms.generative_dataset_root_path, dataset_name)
            datadir_list.append(datadir)
        else:
            raise NotImplementedError

    if args.algorithms.surrogate_data == "dataset":
        # if in args.surrogate_dataset_list:
        train_dl_dict, test_dl_dict, train_ds_dict, test_ds_dict, \
            class_num_dict, train_data_num_dict, test_data_num_dict = load_multiple_centralized_dataset(
                                                                    args=args, 
                                                                    dataset_list=dataset_list, 
                                                                    datadir_list=datadir_list, 
                                                                    resize=image_size, 
                                                                    augmentation="default")
    else:
        raise NotImplementedError

    return train_dl_dict, test_dl_dict, train_ds_dict, test_ds_dict

def load_multiple_centralized_dataset(args, 
                                      dataset_list, 
                                      datadir_list, 
                                      resize=32, 
                                      augmentation="default"): 
    train_dl_dict = {}
    test_dl_dict = {}
    train_ds_dict = {}
    test_ds_dict = {}
    class_num_dict = {}
    train_data_num_dict = {}
    test_data_num_dict = {}

    for i, dataset in enumerate(dataset_list):
        datadir = datadir_list[i]
        generative_dl = Surrogate_DataLoader(args=args, dataset=dataset, augmentation=augmentation,
                                               resize=resize, datadir=datadir)
        train_dl, test_dl, train_data_num, test_data_num, class_num, other_params \
            = generative_dl.load_surrogate_data()

        train_dl_dict[dataset] = train_dl
        test_dl_dict[dataset] = test_dl
        train_ds_dict[dataset] = other_params["train_ds"]
        test_ds_dict[dataset] = other_params["test_ds"]
        class_num_dict[dataset] = class_num
        train_data_num_dict[dataset] = train_data_num
        test_data_num_dict[dataset] = test_data_num

    return train_dl_dict, test_dl_dict, train_ds_dict, test_ds_dict, \
        class_num_dict, train_data_num_dict, test_data_num_dict

def data_transforms_surrogate(resize=None, augmentation="default", dataset_type="full_dataset",
                            image_resolution=32):

    train_transform = transforms.Compose([])
    test_transform = None

    if "grayscale" in augmentation:
        train_transform.transforms.append(
            torchvision.transforms.Grayscale(num_output_channels=1)
        )
        GENERETIVE_MEAN = (0.5)
        GENERETIVE_STD = (0.25)
    else:
        GENERETIVE_MEAN = (0.5, 0.5, 0.5)
        GENERETIVE_STD = (0.25, 0.25, 0.25)

    image_size = image_resolution
    if resize is not image_size:
        image_size = resize
        train_transform.transforms.append(transforms.Resize(resize))

    if "default" in augmentation:
        # pass
        train_transform.transforms.append(transforms.RandomCrop(image_size, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    elif augmentation == "no":
        pass
    else:
        raise NotImplementedError

    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(GENERETIVE_MEAN, GENERETIVE_STD))

    return GENERETIVE_MEAN, GENERETIVE_STD, train_transform, test_transform

class SurrogateDataset(data.Dataset):
    def __init__(self, args, dataset_name="style_GAN_init", datadir="./data",
            dataidxs=None,
            train=True, transform=None, target_transform=None,
            load_in_memory=False,
            image_resolution=32):

        self.args = args
        self.dataset_name = dataset_name

        if dataset_name in ["c10", "c100"]:
            self.labeled = True
        else:
            raise NotImplementedError

        self.image_resolution = image_resolution

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.loader = default_loader
        if self.train:
            self.datadir = os.path.join(datadir, 'train')
        else:
            self.datadir = os.path.join(datadir, 'val')

        self.all_data, self.data_local_num_dict, self.net_dataidx_map = self.__getdatasets__()
        self.initial_local_data()

    def shuffle_data(self):
        # self.local_data = random.shuffle(self.local_data)
        random.shuffle(self.all_data)
        random.shuffle(self.local_data)

    def initial_local_data(self):
        if self.dataidxs == None:
            self.local_data = self.all_data
        elif type(self.dataidxs) == int:
            if self.alpha is not None:
                self.local_data = self.all_data[self.net_dataidx_map[self.dataidxs]]
            else:
                (begin, end) = self.net_dataidx_map[self.dataidxs]
                self.local_data = self.all_data[begin: end]
        else:
            # This is only suitable when not do dirichlet sampling
            assert self.alpha is None
            self.local_data = []
            for idxs in self.dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin: end]

        # self.data_num = sum(list(self.data_local_num_dict.values()))
        self.data_num = len(self.local_data)

    def __getdatasets__(self):
        # all_data = datasets.ImageFolder(datadir, self.transform, self.target_transform)

        classes, class_to_idx = find_classes(self.datadir)
        self.classes = classes
        self.class_num = len(self.classes)
        self.class_to_idx = class_to_idx
        all_data, data_local_num_dict, net_dataidx_map = make_dataset(
            self.datadir, class_to_idx, IMG_EXTENSIONS,
            num_classes=1000, labeled=self.labeled)
        if len(all_data) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.datadir + "\n"
                "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))
        return all_data, data_local_num_dict, net_dataidx_map


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """

        path, target = self.local_data[index]
        img = self.loader(path)
        # logging.info(f"Before transform generative img.size: {img.size}")
        if self.transform is not None:
            img = self.transform(img)
        # logging.info(f"generative img.shape: {img.shape}")

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.local_data)


GENERATIVE_DATASET_LIST = ["c10", "c100"]

class Surrogate_DataLoader(object):

    full_data_obj_dict = {
        "c10": SurrogateDataset,
        "c100": SurrogateDataset,
    } 

    sub_data_obj_dict = {
        "c10": SurrogateDataset,
        "c100": SurrogateDataset,
    } 

    transform_dict = {
        "c10": data_transforms_surrogate,
        "c100": data_transforms_surrogate,
    }

    num_classes_dict = {
        "c10": 10,
        "c100": 100,
    }


    image_resolution_dict = {
        "c10": 32,
        "c100": 32,
    }


    def __init__(self, args, dataset, augmentation, resize, datadir):
        self.args = args
        self.dataset = dataset
        self.augmentation = augmentation
        self.resize = resize
        self.datadir = datadir
        self.batch_size = self.args.algorithms.surrogate_dataset_batch_size
        self.num_workers = self.args.algorithms.dataloader_workers

        self.init_dataset_obj()

    def init_dataset_obj(self):
        self.full_data_obj = Surrogate_DataLoader.full_data_obj_dict[self.dataset]
        self.sub_data_obj = Surrogate_DataLoader.sub_data_obj_dict[self.dataset]
        self.get_transform_func = Surrogate_DataLoader.transform_dict[self.dataset]
        self.class_num = Surrogate_DataLoader.num_classes_dict[self.dataset]
        self.image_resolution = Surrogate_DataLoader.image_resolution_dict[self.dataset]



    def get_transform(self, resize, augmentation, dataset_type="full_dataset", image_resolution=32):
        MEAN, STD, train_transform, test_transform = \
            self.get_transform_func(
                resize=resize, augmentation=augmentation, dataset_type=dataset_type, image_resolution=image_resolution)
        return MEAN, STD, train_transform, test_transform
    
    def load_full_data(self):
        # For cifar10, cifar100, SVHN, FMNIST
        MEAN, STD, train_transform, test_transform = self.get_transform(
            self.resize, self.augmentation, "full_dataset", self.image_resolution)

        logging.debug(f"Train_transform is {train_transform} Test_transform is {test_transform}")
        train_ds = self.full_data_obj(self.args, dataset_name=self.dataset, datadir=self.datadir,
                dataidxs=None,
                train=True, transform=train_transform, target_transform=None,
                load_in_memory=False,
                image_resolution=self.image_resolution)

        test_ds = []

        return train_ds, test_ds


    def load_centralized_data(self):
        self.train_ds, self.test_ds = self.load_full_data()
        self.train_data_num = len(self.train_ds)
        self.test_data_num = len(self.test_ds)
        self.train_dl, self.test_dl = self.get_dataloader(
                self.train_ds, self.test_ds,
                shuffle=True, drop_last=True, train_sampler=None, num_workers=self.num_workers)
    
    def load_surrogate_data(self):
        self.load_centralized_data()
        self.other_params = dict()
        self.other_params["train_ds"] = self.train_ds
        self.other_params["test_ds"] = self.test_ds
        return self.train_dl, self.test_dl, self.train_data_num, self.test_data_num, self.class_num, self.other_params


    def get_dataloader(self, train_ds, test_ds, shuffle=True, drop_last=False, train_sampler=None, num_workers=1):
        # logging.info(f"shuffle: {shuffle}, drop_last:{drop_last}, train_sampler:{train_sampler} ")
        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=shuffle,
                                drop_last=drop_last, sampler=train_sampler, num_workers=num_workers)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.batch_size, shuffle=False,
                                drop_last=False, num_workers=num_workers)
        return train_dl, test_dl
    
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        pass
def find_classes(dir, labeled=True):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    if labeled:
        class_to_idx = {classes[i]: i for i in range(len(classes))}
    else:
        class_to_idx = {classes[i]: 0 for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, extensions, num_classes=1000, labeled=True):
    images = []

    data_local_num_dict = dict()
    data_local_num_dict[0] = 0
    net_dataidx_map = dict()
    sum_temp = 0
    dir = os.path.expanduser(dir)

    i_target = 0 
    for target in sorted(os.listdir(dir)):
        if not (i_target < num_classes):
            break
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        target_num = 0
        if labeled:
            label = class_to_idx[target]
        else:
            label = 0
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    # item = (path, class_to_idx[target])
                    item = (path, label)
                    images.append(item)
                    target_num += 1

        # net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
        # data_local_num_dict[class_to_idx[target]] = target_num

        if labeled:
            net_dataidx_map[label] = (sum_temp, sum_temp + target_num)
            data_local_num_dict[label] = target_num
        else:
            net_dataidx_map[label] = (0, sum_temp + target_num)
            data_local_num_dict[label] += target_num
        sum_temp += target_num
        i_target += 1

    assert len(images) == sum_temp
    return images, data_local_num_dict, net_dataidx_map

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)
