import logging
import copy
import numpy as np



def partition_data(y_train, 
                   client_number, 
                   partition_method, 
                   class_num=10, 
                   partition_alpha=None,
                   seed=0):
    if partition_method in ["homo", "iid"]:
        total_num = len(y_train)
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, client_number)
        net_dataidx_map = {i: batch_idxs[i] for i in range(client_number)}
    elif partition_method == "hetero" and partition_alpha is not None:
        min_size = 0
        min_require_size = 10

        K = class_num
        N = y_train.shape[0]
        
        logging.info("partition num for clients = " + str(N))
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(client_number)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(partition_alpha, client_number))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / client_number) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(client_number):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    elif partition_method > "noniid-#label0" and partition_method <= "noniid-#label9":
        num = eval(partition_method[13:])
        
        times=[0 for i in range(class_num)]
        contain=[]
        for i in range(client_number):
            current=[i%class_num]
            times[i%class_num]+=1
            j=1
            while (j<num):
                ind=np.random.randint(0,class_num-1)
                if (ind not in current):
                    j=j+1
                    current.append(ind)
                    times[ind]+=1
            contain.append(current)
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(client_number)}
        for i in range(class_num):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k,times[i])
            ids=0
            for j in range(client_number):
                if i in contain[j]:
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                    ids+=1
        for i in range(client_number):
            net_dataidx_map[i] = net_dataidx_map[i].tolist()
    traindata_cls_counts = record_client_data_stats(y_train, net_dataidx_map)
    return net_dataidx_map, traindata_cls_counts

def record_y_distribution(y):
    unq, unq_cnt = np.unique(y, return_counts=True)
    tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
    return tmp

def record_client_data_stats(y_train, 
                             net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts

def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor):
    img_max = len(list_label2indices_train) / num_classes
    img_num_per_cls = []
    for _classes_idx in range(num_classes):
        num = img_max * (imb_factor**(_classes_idx / (num_classes - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls

def classify_label(y, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, label in enumerate(y):
        list1[label].append(idx)
    return list1

def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type):
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
    img_num_list = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor)

    list_clients_indices = []
    classes = list(range(num_classes))
    for _class, _img_num in zip(classes, img_num_list):
        indices = list_label2indices_train[_class]
        np.random.shuffle(indices)
        idx = indices[:_img_num]
        list_clients_indices.append(idx)
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    logging.info('All num_data_train in long tail distribution')
    logging.info(len(num_list_clients_indices))
    return img_num_list, list_clients_indices

def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res

def get_train_batch_data(train_local_iter_dict, dataset_name, train_local, batch_size, drop_last=True):
    try:
        train_batch_data = next(train_local_iter_dict)
        # logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
        if len(train_batch_data[0]) < batch_size:
            if drop_last:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), batch_size))
                logging.debug("Using Drop Last, reinitialize loader.")
                train_local_iter_dict = iter(train_local)
                train_batch_data = next(train_local_iter_dict)
            else:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), batch_size))

            # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
            # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
    except:
        train_local_iter_dict = iter(train_local)
        train_batch_data = next(train_local_iter_dict)
    return train_batch_data