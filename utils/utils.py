import copy
import numpy as np
import random
import torch 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging

import wandb
from data_preprocessing.own_transforms import get_stransform
from data_preprocessing.personalized_dataset import Dataset_Relabel


def loss_gamma(predictions,labels,model,delta_control_models,lamb):
    sca_loss = 0
    for name, param in model.named_parameters():
        if name in delta_control_models:
            sca_loss += torch.sum(param * delta_control_models[name]) * lamb
    ce_loss = torch.nn.functional.cross_entropy(predictions,labels,reduction='mean')
    loss = ce_loss + sca_loss
    return ce_loss, (ce_loss,sca_loss)

def proximal_loss(global_params, local_model, device):
    proximal_term = 0.0
    for name, local_param in local_model.named_parameters():
        global_param = global_params[name].to(device).to(torch.float32)
        proximal_term += (global_param - local_param).norm(2)
    return proximal_term

def compare_state_dicts(sd1, sd2):
    for key in sd1:
        if key not in sd2:
            return False
        if not torch.equal(copy.deepcopy(sd1[key]).cpu(), copy.deepcopy(sd2[key]).cpu(),):
            return False
    for key in sd2:
        if key not in sd1:
            return False
    return True

def pseudo_label_batch_wise(model, data, device):
    pass


def compute_relabel_class_wise_acc(num_classes, relabel_labels, relabel_gt_labels):
    """
    Args:
        num_classes (`int`):
             the number of classes
        relabel_labels (`np.arrat/torch.Tensor` of shape `(num_samples,)`):
            all relabeled data samples
        relabel_gt_labels (`np.arrat/torch.Tensor` of shape `(num_samples,)`):
            its corresponding ground truth label
    Return:
        class_acc (`np.arrat/torch.Tensor` of shape `(num_samples,)`)
    """
    sum_correct = (relabel_labels == relabel_gt_labels).sum().item()
    acc = sum_correct / len(relabel_labels)
    class_acc = { cls:0 for cls in range(num_classes)}
    for idx in range(len(relabel_labels)):
        if relabel_labels[idx] == relabel_gt_labels[idx]:
            class_acc[relabel_labels[idx]] += 1
        
    return acc, class_acc

def pseudo_label_dataset_wise(ulb_data, 
                              gt_labels, 
                              model, 
                              dataset_name, 
                              device, 
                              num_classes=10, 
                              threshold_type='all', 
                              threshold=0.9):
    """
    Args:
        ulb_data (`torch.Tensor` of shape `(num_samples, C, H, W)`):
            the unlabeled data
        gt_labels (`torch.Tensor` of shape `(num_samples,)`):
            the ground truth labels of the unlabeled data
        model (`torch.nn.Module`):
            the model to be used
        dataset_name (`str`):
            the name of the dataset
        device (`torch.device`):
            the device to be used
        num_classes (`int`):
            the number of classes
        threshold_type (`str`):
            the type of threshold
        threshold (`float`):
            the threshold
    Return:
        relabeled_data (`torch.Tensor` of shape `(num_samples, C, H, W)`):
            the relabeled data
        relabeled_labels (`torch.Tensor` of shape `(num_samples,)`):
            the relabeled labels
        acc (`float`):
            the accuracy
        class_acc (`dict`):
            the class accuracy
    """
    train_trans = get_stransform(dataset_name, train=True)
    label_ds = Dataset_Relabel(data=ulb_data,
                                  targets=gt_labels,
                                  ulb=False,
                                  dataset=dataset_name,
                                  transform=train_trans)
    label_dl = torch.utils.data.DataLoader(dataset=label_ds,
                                           batch_size=64, shuffle=False,
                                          drop_last=False)

    model.eval()
    model.to(device)

    collect_confident_data = []
    collect_confident_labels = []
    collect_gt_confident_labels = []
    
    if threshold_type == 'all':
        with torch.no_grad():
            for batch_idx, (ori_data, weak_data, labels) in enumerate(label_dl):
                weak_data = weak_data.to(device)
                logits = model(weak_data)
                logits = logits.detach()
                probs = F.softmax(logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1)

                confident_idx = torch.where(max_probs >= threshold)[0]
                
                confident_data = ori_data[confident_idx]
                gt_confident_labels = labels[confident_idx]
                confident_labels = preds[confident_idx]

                collect_confident_data.append(confident_data)
                collect_confident_labels.append(confident_labels)
                collect_gt_confident_labels.append(gt_confident_labels)
            relabeled_data = torch.cat(collect_confident_data) if collect_confident_data else torch.empty(0)
            relabeled_labels = torch.cat(collect_confident_labels) if collect_confident_labels else torch.empty(0)
            relabeled_gt_labels = torch.cat(collect_gt_confident_labels) if collect_confident_labels else torch.empty(0)
            acc, class_acc = compute_relabel_class_wise_acc(num_classes=num_classes,
                                                       relabel_labels=relabeled_labels,
                                                       relabel_gt_labels=relabeled_gt_labels)

        return relabeled_data, relabeled_labels, acc, class_acc
    elif threshold_type == 'class_wise':
        with torch.no_grad():
            for batch_idx, (ori_data, weak_data, labels) in enumerate(label_dl):
                weak_data = weak_data.to(device)
                logits = model(weak_data)
                logits = logits.detach()
                probs = F.softmax(logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                sample_wise_confident_threshold = torch.zeros(preds.shape)

                # sample_wise_confident_threshold `(batch_size, )`
                for i in range(len(preds)):
                    sample_wise_confident_threshold[i] =  threshold[preds[i]]
                
                confident_idx = torch.where(max_probs >= sample_wise_confident_threshold)[0]

                confident_data = ori_data[confident_idx]
                gt_confident_labels = labels[confident_idx]
                confident_labels = preds[confident_idx]

                collect_confident_data.append(confident_data)
                collect_confident_labels.append(confident_labels)
                collect_gt_confident_labels.append(gt_confident_labels)
            relabeled_data = torch.cat(collect_confident_data) if collect_confident_data else torch.empty(0)
            relabeled_labels = torch.cat(collect_confident_labels) if collect_confident_labels else torch.empty(0)
            relabeled_gt_labels = torch.cat(collect_gt_confident_labels) if collect_confident_labels else torch.empty(0)
            acc, class_acc = compute_relabel_class_wise_acc(num_classes=num_classes,
                                                       relabel_labels=relabeled_labels,
                                                       relabel_gt_labels=relabeled_gt_labels)

        return relabeled_data, relabeled_labels, acc, class_acc

def get_data_distribution(labels, num_classes):
    """
    Get the distribution of the labels
    Args:
        labels (`np.array` of shape `(N,)`):
            the labels of the data
        num_classes (`int`):
            the number of classes
    Return:
        distribution (`np.array` of shape `(num_classes,)`):
            the distribution of the labels
    """
    distribution = np.zeros(num_classes)
    for i in labels:
        distribution[i] = (labels == i).float().sum()
    return distribution

def merge_lb_ulb_distribution(lb_distribution, ulb_distribution):
    """
    Merge the distribution of the labeled data and the unlabeled data
    Args:
        lb_distribution (`np.array` of shape `(num_classes,)`):
            the distribution of the labeled data
        ulb_distribution (`np.array` of shape `(num_classes,)`):
            the distribution of the unlabeled data
    Return:
        merged_distribution (`np.array` of shape `(num_classes,)`):
            the merged distribution of the labeled data and the unlabeled data
    """
    if lb_distribution.shape != ulb_distribution.shape:
        raise ValueError("The distribution of the labeled data and the unlabeled data must have the same shape")
    else:
        total_distribution = lb_distribution + ulb_distribution
        return total_distribution
    

def zero_weights(model, type='model'):
    with torch.no_grad():
        if type == 'model':
            # 处理参数
            for name, param in model.named_parameters():
                param.data.zero_()
            # 处理缓冲区（包括 running_mean 和 running_var）
            for name, buffer in model.named_buffers():
                buffer.zero_()
        elif type == 'params':
            for param in model.values():
                param.data.zero_()

def check_remove_batch_norm_stats_info(model_params:dict):
    if model_params is not None:
        new_model_params = copy.deepcopy(model_params)
        for name, param in new_model_params.items():
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                model_params.pop(name)
        return model_params

def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)
def set_random(seed):
    torch.manual_seed(seed)      
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed)         
    random.seed(seed)           
    torch.backends.cudnn.benchmark = False   
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def compute_fim_condition_number(model):
    """
    Compute the condition number of the Fisher Information Matrix
    """
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    fim = torch.outer(grads, grads)
    eig_vals = torch.linalg.eigvals(fim + 1e-6 * torch.eye(fim.size(0)))
    condition_number = torch.max(eig_vals) / torch.min(eig_vals)
    return fim, condition_number

def compute_fim_condition_number_sample(model, max_size=5000):
    """
    在不影响原始梯度的情况下计算FIM条件数
    """
    grads = []
    with torch.no_grad():  # 使用no_grad避免创建计算图
        for param in model.parameters():
            if param.grad is not None:
                # 创建梯度的副本并移到CPU
                grad_clone = param.grad.clone().detach().cpu().float().view(-1)
                if grad_clone.size(0) > max_size:
                    idx = torch.randperm(grad_clone.size(0))[:max_size]
                    grad_clone = grad_clone[idx]
                grads.append(grad_clone)
    
    # 在CPU上进行后续计算
    grads = torch.cat(grads)
    if grads.size(0) > max_size:
        idx = torch.randperm(grads.size(0))[:max_size]
        grads = grads[idx]
    
    # 分批计算FIM以节省内存
    fim = torch.zeros((grads.size(0), grads.size(0)), device='cpu')
    batch_size = 100
    for i in range(0, grads.size(0), batch_size):
        end = min(i + batch_size, grads.size(0))
        fim[i:end] = torch.outer(grads[i:end], grads)
    
    # 计算条件数
    eig_vals = torch.linalg.eigvals(fim + 1e-6 * torch.eye(fim.size(0), device='cpu'))
    condition_number = torch.max(eig_vals.abs()) / torch.min(eig_vals.abs())
    
    del fim, grads  # 显式释放内存
    torch.cuda.empty_cache()  # 清理GPU缓存
    
    return condition_number.item()  # 只返回条件数的标量值


def compute_fim_condition_number_by_layer(model):
    """
    分层计算FIM条件数并记录到wandb
    """
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            print(f"Processing layer: {name}")
            if param.grad is not None:
                layer_stats = {}
                # 创建当前层梯度的副本并移到CPU
                grad_clone = param.grad.clone().detach().cpu().float().view(-1)
                
                # 计算当前层的FIM
                fim = torch.outer(grad_clone, grad_clone)
                print("F norm:{} at {} layer ".format(torch.norm(fim, p='fro'), name))
                rank = torch.linalg.matrix_rank(fim)
                print("F rank:{} at {} layer ".format(rank, name))
                # 计算条件数
                # eig_vals = torch.linalg.eigvals(fim + 1e-6 * torch.eye(fim.size(0)))
                # condition_number = torch.max(eig_vals.abs()) / torch.min(eig_vals.abs())
                # # condition_number = torch.linalg.cond(fim + 1e-6 * torch.eye(fim.size(0)).to(fim.device))
                # fim = fim.cpu().numpy()
                # # 存储结果
                # layer_name = name.replace('.', '/')  # 替换点号为斜杠，避免wandb的键名问题
                # layer_stats[f"condition_number/{layer_name}"] = condition_number.item()
                
                # for key in layer_stats:
                #     wandb.log({key: layer_stats[key]})
                # wandb.log({"{}_fim_histogram".format(layer_name): wandb.Histogram(fim)})

                
                # 清理内存
                del fim
                del grad_clone            
    torch.cuda.empty_cache()

def count_batch_distribution(batch_label,num_classes):
    """
    Count the distribution of the batch labels
    """
    distribution = torch.zeros(num_classes)
    for i in batch_label:
        distribution[i] += 1
    return distribution