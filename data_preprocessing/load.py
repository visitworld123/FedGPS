import logging
import numpy as np
import pickle
import copy
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms as transforms
from torchvision.datasets import (
     CIFAR10, 
     CIFAR100, 
     SVHN, 
     FashionMNIST
     )

from data_preprocessing.utils import (
    partition_data,
    record_y_distribution,
    classify_label,
    train_long_tail
    )
from data_preprocessing.load_FLAIR_data import load_FLAIR_data

full_data_obj_dict = {
        "CIFAR10": CIFAR10,
        "CIFAR100": CIFAR100,
        "SVHN": SVHN,
        "FMNIST": FashionMNIST
    } 

def long_tail_simulation(X_train, y_train, num_classes, imb_factor):
    list_label2indices = classify_label(y_train, num_classes)
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), num_classes,
                                                    imb_factor=imb_factor, imb_type='exp')
    idx = list(itertools.chain(*list_label2indices_train_new))
    return X_train[idx], y_train[idx]

def load_full_data(dataset,  
                   datadir):

        if dataset == "SVHN":
            train_ds = full_data_obj_dict[dataset](datadir,  "train", download=True, transform=transforms.Compose([transforms.ToTensor()]), target_transform=None)
            test_ds = full_data_obj_dict[dataset](datadir,  "test", download=True, transform=transforms.Compose([transforms.ToTensor()]), target_transform=None)
        else:
            train_ds = full_data_obj_dict[dataset](datadir,  train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            test_ds = full_data_obj_dict[dataset](datadir,  train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

        X_train = train_ds.data
        X_test = test_ds.data

        if dataset in ["fmnist"]:
            y_train = train_ds.targets.data
            y_test = test_ds.targets.data
        elif dataset in ["SVHN"]:
            y_train = train_ds.labels
            y_test = test_ds.labels
        else:
            y_train = train_ds.targets
            y_test = test_ds.targets
        
        y_train_np = np.array(y_train)
        y_dis = record_y_distribution(y_train_np)
        y_test_np = np.array(y_test)

        return X_train, y_train_np, X_test, y_test_np

def FSSL_labels_at_server(dataset,
                          datadir,
                          supervised_num_at_server, 
                          client_number, 
                          partition_method, 
                          class_num=10,
                          partition_alpha=None,
                          seed=0):
    data_split_save_dict={}
    X_train_total, y_train_total, X_test, y_test_np = load_full_data(dataset, datadir)
    X_test = np.array(X_test)
    num_supervised_per_class = supervised_num_at_server // class_num
    supervised_idx = []

    for i in range(class_num):
        idx = np.where(y_train_total == i)[0]
        idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
        supervised_idx.extend(idx)
    idx = list(range(len(y_train_total)))
    unsupervised_idx = list(set(idx) - set(supervised_idx))
    data_split_save_dict['server_sup'] = supervised_idx
    server_X_train = np.array([X_train_total[s] for s in supervised_idx])
    server_y_train = np.array([y_train_total[s] for s in supervised_idx])
    rest_X_train = np.array([X_train_total[s] for s in unsupervised_idx])
    rest_y_train = np.array([y_train_total[s] for s in unsupervised_idx])

    client_dataidx_map, traindata_cls_counts = partition_data(rest_y_train, 
                                                            client_number, 
                                                            partition_method, 
                                                            class_num=class_num, 
                                                            partition_alpha=partition_alpha,
                                                            seed=seed)


    client_X_train_dict = {}
    client_y_train_dict = {}
    client_sample_dict = {}
    for client_num in client_dataidx_map.keys():
        client_X_train_dict[client_num] = np.array([rest_X_train[i] for i in client_dataidx_map[client_num]])
        client_y_train_dict[client_num] = np.array([rest_y_train[i] for i in client_dataidx_map[client_num]])
        client_sample_dict[client_num] = len(client_y_train_dict[client_num])
    data_split_save_dict['client_unsup_map'] = client_dataidx_map

    with open("{}_{}_{}_{}_{}.pkl".format(dataset,partition_method,partition_alpha,client_number,seed), "wb") as file:
        pickle.dump(data_split_save_dict, file)
    logging.info('Data statistics' )
    for client_num in traindata_cls_counts.keys():
        logging.info('Client {}, data count = {}, detial class num = {}'.format(client_num, client_sample_dict[client_num], traindata_cls_counts[client_num]))
    return server_X_train, server_y_train,  X_test, y_test_np, client_X_train_dict, client_y_train_dict, client_sample_dict


def split_data_into_clients(dataset,
                            datadir,
                            client_number, 
                            partition_method, 
                            long_tail=False,
                            class_num=10,
                            partition_alpha=None,
                            seed=0,
                            **kwargs):
    if dataset in ['CIFAR100', 'CIFAR10', 'SVHN']:
        data_split_save_dict={}
        X_train_total, y_train_total, X_test, y_test_np = load_full_data(dataset, datadir)
        if long_tail:
            X_train_total, y_train_total = long_tail_simulation(X_train_total, y_train_total, class_num, imb_factor=kwargs['long_tail_imb_factor'])
        y_dis = record_y_distribution(y_train_total)
        logging.info("The total label distribution of simulated dataset")
        logging.info(y_dis)
        X_test = np.array(X_test)

        client_dataidx_map, traindata_cls_counts = partition_data(y_train_total, 
                                                                  client_number, 
                                                                  partition_method, 
                                                                  class_num=class_num, 
                                                                  partition_alpha=partition_alpha,
                                                                  seed=seed)

        client_X_train_dict = {}
        client_y_train_dict = {}
        client_sample_dict = {}
        for client_num in client_dataidx_map.keys():
            client_X_train_dict[client_num] = np.array([X_train_total[i] for i in client_dataidx_map[client_num]])
            client_y_train_dict[client_num] = np.array([y_train_total[i] for i in client_dataidx_map[client_num]])
            client_sample_dict[client_num] = len(client_y_train_dict[client_num])
        data_split_save_dict['client_unsup_map'] = client_dataidx_map

        # with open("{}_{}_{}_{}_{}.pkl".format(dataset,partition_method,partition_alpha,client_number,seed), "wb") as file:
        #     pickle.dump(data_split_save_dict, file)
        logging.info('Data statistics' )
        for client_num in traindata_cls_counts.keys():
            logging.info('Client {}, data count = {}, detial class num = {}'.format(client_num, client_sample_dict[client_num], traindata_cls_counts[client_num]))
        return client_X_train_dict, client_y_train_dict, X_test, y_test_np
    
    elif dataset in 'FLAIR':
        client_X_train_dict, client_y_train_dict, X_test, y_test_np = load_FLAIR_data(datadir, 
                                                                                      client_number)
        return client_X_train_dict, client_y_train_dict, X_test, y_test_np
    
    
def draw_heatmap(data, name):

# Determine the number of clients and classes
    num_clients = len(data)
    num_classes = max(max(data[client].keys()) for client in data) + 1  # Assumes class indices start from 0

# Create a matrix to store the data (rows: classes, columns: clients)
    matrix = np.zeros((num_classes, num_clients))

# Fill the matrix with data
    for client_idx in data:
        for class_idx, num in data[client_idx].items():
            matrix[class_idx, client_idx] = num

# 设置图形风格
    plt.style.use('default')  # 使用默认风格以获得更清晰的显示效果
    plt.rcParams['font.family'] = 'DejaVu Serif'  # 设置全局字体为 DejaVu Serif

# 创建图形，设置合适的宽高比
    plt.figure(figsize=(8, 8), dpi=800)  # 调整宽高比以适应数据分布

# Create y-axis labels and ticks - only show every 10th
    yticks = []
    yticklabels = []
    for i in range(num_classes):
        if i % 1 == 0:
            yticks.append(i)
            yticklabels.append(f"{i}")

    # 创建热图，使用更适合的颜色方案
    ax = sns.heatmap(matrix, 
                    cmap="YlOrRd",  # 使用更醒目的颜色方案
                    cbar_kws={'label': 'Sample Count'},  # 添加颜色条标签
                    yticklabels=False,
                    xticklabels=[f"{i}" for i in range(num_clients)],
                    square=False)  # 关闭正方形设置以适应数据形状

    # 设置y轴刻度和标签
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontname='DejaVu Serif')

    # 自定义外观
    plt.xticks(rotation=0, fontsize=10, fontname='DejaVu Serif')
    plt.yticks(fontsize=10, fontname='DejaVu Serif')

    # 添加标签
    plt.xlabel("Client ID", fontsize=12, labelpad=10, fontname='DejaVu Serif')
    plt.ylabel("Label ID", fontsize=12, labelpad=10, fontname='DejaVu Serif')
    plt.xlabel("Client ID", fontsize=12, labelpad=10, fontname='Times New Roman')
    plt.ylabel("Label ID", fontsize=12, labelpad=10, fontname='Times New Roman')
    plt.title("Data Distribution Across Clients", fontsize=14, pad=20, fontname='Times New Roman')

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(name, dpi=800, bbox_inches='tight', pad_inches=0.1)