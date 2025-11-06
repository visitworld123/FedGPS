import random

import numpy as np

def noisification(true_label, num_classes=10, noise_type="symmetric"):
    if noise_type == "symmetric":
        label_lst = list(range(num_classes))
        label_lst.remove(true_label)
        return random.sample(label_lst, k=1)[0]

    elif noise_type == "pairflip":
        return (true_label - 1) % num_classes
    
def noisification_in_client(true_label, cls_arr, noise_type="symmetric"):
    if noise_type == "symmetric":
        label_lst = list(cls_arr)
        label_lst.remove(true_label)
        return random.sample(label_lst, k=1)[0]

    elif noise_type == "pairflip":
        return cls_arr[np.where(true_label==cls_arr)[0]-1]