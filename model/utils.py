import numpy as np
import logging
import torch
import torch.nn.functional as F
from model.ssl_model import SimCLRModel, BYOLModel, SimSiamModel
from model.ResNet import ResNet18
from model.SemiFed import SemiFed

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def info_nce_loss( features, batch_size, device, n_views=2, temperature=0.07):
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     n_views * self.conf.batch_size, n_views * self.conf.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels

def build_SemiFed_model(args, 
                        num_classes, 
                        model_input_channels, 
                        semi_model, 
                        base_model):
    if base_model == 'resnet18':
        unsup_net = ResNet18(args=args, num_classes=num_classes,
                    input_channels=model_input_channels)
        sup_model = ResNet18(args=args, num_classes=num_classes,
                            input_channels=model_input_channels)
    if semi_model == 'SimCLR':
        unsup_model = SimCLRModel(unsup_net)
    elif semi_model == 'SimSiam':
        unsup_model = SimSiamModel(unsup_net)
    elif semi_model == 'BYOL':
        unsup_model = BYOLModel(unsup_net)
    else:
        raise NotImplementedError(("Semi model '{semi_model}' not implemented."))
    
    model = SemiFed(unsup_model, sup_model, 
                    semi_model=semi_model, 
                    input_channel=model_input_channels, 
                    num_class=num_classes)
    return model


def consistency_loss_fixmatch(logits_w, logits_s, T=1.0, p_cutoff=0.0, use_hard_labels=True):
    
    logits_w = logits_w.detach()
    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(p_cutoff).float()
    select = max_probs.ge(p_cutoff).long()

    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
    else:
        pseudo_label = torch.softmax(logits_w / T, dim=-1)
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask

    return masked_loss.mean(), mask.mean(), select, max_idx.long()

def consistency_loss_freematch(dataset, logits_w, logits_s, time_p, p_model, use_hard_labels=True):

    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    p_cutoff = time_p
    p_model_cutoff = p_model / torch.max(p_model,dim=-1)[0]
    threshold = p_cutoff * p_model_cutoff[max_idx]
    if dataset == 'SVHN':
        threshold = torch.clamp(threshold, min=0.9, max=0.95)
    mask = max_probs.ge(threshold)
    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask.float()
    else:
        pseudo_label = torch.softmax(logits_w / 0.5, dim=-1)
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask.float()
    return masked_loss.mean(), mask


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss