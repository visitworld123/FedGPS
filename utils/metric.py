import copy
import torch
import numpy as np

import torch.nn.functional as F

def accuracy(output, 
             target, 
             topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)

    # How many classes in target and whats the number of them
    unq, unq_cnt = np.unique(target.cpu(), return_counts=True)
    total_class = {int(unq[i]): unq_cnt[i] for i in range(len(unq))}     # dict to record class and corresponding num {class: class_num}

    output = F.softmax(output, dim=1)

    class_acc = {int(unq[i]): 0 for i in range(len(unq))}  # {class: 0}
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)    # output values=[batch_size,maxk]  indices
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    for label, prediction in zip(target, pred.t()):
        if label == prediction[:1]:
            class_acc[int(label)] += 1


    res = []
    correct_num = []
    for k in topk: # (1,5)
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        correct_num.append(copy.deepcopy(correct_k))
        res.append(correct_k.mul_(100.0 / batch_size))
        

    if len(res) == 1:
        return res[0], correct_num[0], class_acc   # res[0] 保存top1的acc，以此类推，topk=(1,5)则res[1]中保存tok[1]即top5的acc
    else:
        return (res[0], res[1], correct[0], pred[0], class_acc)

def top_one_accuracy(outputs, 
                     labels):
    _, pred_labels = torch.max(outputs, 1)
    pred_labels = pred_labels.view(-1)
    correct = torch.sum(torch.eq(pred_labels, labels)).item()
    total = len(labels)

    return correct, total

def MAccuracy(output, target, mask, topk=1):
    if torch.any(mask):
        output = output[mask]
        target = target[mask]
        acc, _, _ = accuracy(output, target)
    else:
        acc = 0
    return acc


def LabelRatio(mask):
    with torch.no_grad():
        lr = mask.float().mean().item()
    return lr