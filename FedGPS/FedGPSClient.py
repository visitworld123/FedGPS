import copy
import logging
import math

import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
from utils.averager import AverageMeter
from utils.record import record
from baseFedAvg.client import Client
from utils.metric import (
    accuracy,
    top_one_accuracy
)
from utils.utils import zero_weights
from utils.optim import FedGPSOptimizer
from baseFedAvg.client import Client
from utils.utils import check_remove_batch_norm_stats_info
from data_preprocessing.utils import get_train_batch_data
from FedGPS.surrogate_dataset import surrogate_load_dataset
from FedGPS.utils import proxy_align_loss
from torch.nn import functional as F
from utils.utils import loss_gamma
class FedGPSClient(Client):

    def __init__(self, 
                 args,
                 device,
                 index,
                 X_train, 
                 y_train, 
                 X_test, 
                 y_test) -> None:
        super().__init__(args,
                device,
                index,
                X_train, 
                y_train, 
                X_test, 
                y_test)
        
        self.role = 'FedGPSClient'
        self.local_control_model = copy.deepcopy(self.model)
        zero_weights(self.local_control_model)
        self.model_delta = OrderedDict()
        self.delta_c = OrderedDict()
        self.grad_pert_corr_delta = OrderedDict()
        self.local_grad_pert_corr_model = self.get_model_params()
        zero_weights(self.local_grad_pert_corr_model, type='params')
        self.local_noise_protos = {}
        self.proto_agg_weights = {}
        self.lr = self.args.optim.lr
    def _initial_setup(self):
        if self.args.algorithms.surrogate:
            if self.args.algorithms.surrogate_label_shift:
                self.model = self._build_model(num_classes=self.args.datasets.num_classes+
                                               self.args.datasets.num_classes)
            else:
                self.model = self._build_model()
        self.opt = self._build_optimizer(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.args.algorithms.local_epochs)
        if self.args.algorithms.surrogate:
            self._surrogate_prepare()

    def get_train_batch_data(self, train_local):
        try:
            train_batch_data = next(self.train_local_iter)
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
               
        except:
            self.train_local_iter = iter(train_local)
            train_batch_data = next(self.train_local_iter)
        return train_batch_data
    def create_noise_dataset_dict(self):
        self.train_generative_dl_dict, self.test_generative_dl_dict, \
        self.train_generative_ds_dict, self.test_generative_ds_dict \
            = surrogate_load_dataset(self.args)
        self.noise_dataset_label_shift = {}
        noise_dataset_label_init = 0
        next_label_shift = noise_dataset_label_init
        for dataset_name in self.train_generative_dl_dict.keys():
            self.noise_dataset_label_shift[dataset_name] = next_label_shift
            next_label_shift += next_label_shift + self.train_generative_ds_dict[dataset_name].class_num

    def _surrogate_prepare(self):
        if self.args.algorithms.surrogate_feat_align:
            self.surrogate_mapping_matrix = torch.rand(
                        self.args.algorithms.model_feature_dim, self.args.algorithms.model_feature_dim)
            self.proxy_align_loss = proxy_align_loss(
                    inter_domain_mapping=self.args.algorithms.surrogate_inter_domain_mapping,
                    inter_domain_class_match=self.args.algorithms.surrogate_class_match,
                    noise_feat_detach=self.args.algorithms.surrogate_feat_detach,
                    noise_contrastive=self.args.algorithms.surrogate_noise_contrastive,
                    inter_domain_mapping_matrix=self.surrogate_mapping_matrix,
                    inter_domain_weight=self.args.algorithms.surrogate_feat_align_inter_domain_weight,
                    inter_class_weight=self.args.algorithms.surrogate_feat_align_inter_cls_weight,
                    noise_supcon_weight=self.args.algorithms.surrogate_noise_supcon_weight,
                    noise_label_shift=self.args.datasets.num_classes,
                    device=self.device)
        if self.args.algorithms.surrogate_data == 'dataset':
            self.create_noise_dataset_dict()
        
    def get_params(self, model, type='model', with_nograd=True):
        if type == 'model':
            if with_nograd:
                return {k: copy.deepcopy(val.cpu())
                        for k, val in model.state_dict().items()}
            else:
                return {k: copy.deepcopy(val.cpu())
                        for k, val in model.named_parameters()}
        elif type == 'param':
            return {k: copy.deepcopy(val.cpu())
                    for k, val in model.items()}
        else:
            raise ValueError('Invalid type: {}'.format(type))

    def generate_noise_data(self, noise_label_style="extra"):

        noise_label_shift = 0

        if noise_label_style == "extra":
            noise_label_shift = self.args.datasets.num_classes
            chunk_num = self.args.algorithms.surrogate_num
            chunk_size = self.args.bs // chunk_num
            # chunks = np.ones(chunk_num)* chunk_size
            chunks = [chunk_size] * chunk_num
            for i in range(self.args.bs - chunk_num * chunk_size):
                chunks[i] += 1
        else:
            raise NotImplementedError

        if self.args.algorithms.surrogate_data == "dataset" and self.args.algorithms.surrogate_label_from == "dataset":

            noise_data_list = []
            noise_data_labels = []
            # In order to implement traverse the extra datasets, automatically generate iterator.
            for dataset_name, train_generative_dl in self.train_generative_dl_dict.items():
                generative_iter = iter(train_generative_dl)
                train_batch_data = get_train_batch_data(generative_iter, dataset_name,
                    train_generative_dl, batch_size=self.args.bs / len(self.train_generative_dl_dict))
                logging.debug(f"id(generative_iter) : {id(generative_iter)}")
                data, label = train_batch_data
                noise_data_list.append(data)
                label_shift = self.noise_dataset_label_shift[dataset_name] + noise_label_shift
                if self.args.algorithms.surrogate_label_shift:
                    noise_data_labels.append(label + label_shift)
                else:
                    noise_data_labels.append(label)
            noise_data = torch.cat(noise_data_list).to(self.device)
            labels = torch.cat(noise_data_labels).to(self.device)
        else:
            raise NotImplementedError

        return noise_data, labels

    def _build_optimizer(self, 
                         params_to_optimizer, 
                         lr=None,
                         weight_decay=None,
                         momentum=None):

        if lr is None:
            lr = self.args.optim.lr
        if weight_decay is None:
            weight_decay = self.args.optim.wd
        if momentum is None:
            momentum = self.args.optim.momentum
        if momentum < 1.0:
            self.base_opt = torch.optim.SGD(params_to_optimizer,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    nesterov=self.args.optim.nesterov)
        else:
            self.base_opt =torch.optim.SGD(params_to_optimizer,
                                    lr=self.args.optim.lr,
                                    weight_decay=self.args.optim.wd)
        optimizer = FedGPSOptimizer(self.model.parameters(),
                            base_optimizer=self.base_opt,
                            rho=self.args.algorithms.rho)
        
        decay_factor = 0.998
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: decay_factor
        )
        
    
        return optimizer
    
    def set_params(self, model, params, with_nograd=True):
        if with_nograd:
            model.load_state_dict(params)
        else:
            for name, param in model.named_parameters():
                param.data.copy_(params[name])

    def client_update_local_info(self, download_info):

        if 'GLOBAL_MODEL_PARAM' in download_info:
            logging.info("Client {} update local model with global model".format(self.index))
            self.set_params(self.model,
                            copy.deepcopy(download_info['GLOBAL_MODEL_PARAM']), 
                            with_nograd=self.args.share_with_nongrad)
            self.global_model_params = download_info['GLOBAL_MODEL_PARAM']
        else:
            raise ValueError('Client does not receive global model')
    
        if 'GLOBAL_UPDATE_PARAM' in download_info :

            logging.info("Client {} update global update model locally".format(self.index))
            self.global_update = copy.deepcopy(download_info['GLOBAL_UPDATE_PARAM'])
        else:
            raise ValueError('Client does not receive global update model')
        if 'GLOBAL_EMA_UPDATE_PARAM' in download_info:
            logging.info("Client {} update global ema update model locally".format(self.index))
            self.global_ema_update = copy.deepcopy(download_info['GLOBAL_EMA_UPDATE_PARAM'])
        else:
            raise ValueError('Client does not receive global ema update model')
    
        if 'GLOBAL_NOISE_PROTOS_PARAM' in download_info:
            logging.info("Client {} update local noise protos with global noise protos".format(self.index))
            self.global_noise_protos = copy.deepcopy(download_info['GLOBAL_NOISE_PROTOS_PARAM'])
        else:
            raise ValueError('Client does not receive global noise protos')

    def train(self,
              round,
              train_epoch,
              dataloader,
              **kwargs):
        logging.info("-------------------FedGPS local train-------------------")      
        self.model.to(self.device)
        self.model.train()
        loss_func = loss_gamma
        ce = torch.nn.CrossEntropyLoss()
        triplet_loss = TripletLoss()
        if self.args.algorithms.pre_selected_client_use_minus:
            if kwargs['pre_flag']:
                logging.info("use minus")
                use_minus = True
            else:
                use_minus = False
        else:
            if not kwargs['pre_flag']:
                logging.info("use minus")
                use_minus = True
            else:
                use_minus = False

        if self.global_update !=None and len(self.model_delta) > 0:
            g_update = []
            for key, param in self.global_update.items():
                if 'running' in key or 'num_batches_tracked' in key:
                    continue
                if self.args.algorithms.same_symbol:
                    # non-self gradient 
                        if use_minus:
                            tmp_info = ((self.args.client_number_per_round  * param.to(self.device) - self.model_delta[key].to(self.device)) / (self.args.client_number_per_round - 1)).to(self.device)
                        else:
                            tmp_info = param.to(self.device)
                g_update.append(tmp_info)
        else:
            logging.info("No global update model")
            g_update = None


        for epoch in range(train_epoch):
            # self.base_opt.param_groups[0]['lr'] = self.lr
            logging.info('=> Training Epoch #%d, LR=%.4f' % (epoch, self.base_opt.param_groups[0]['lr']))
            train_acc_avg = AverageMeter()
            train_loss_avg = AverageMeter()
            train_ce_loss_avg = AverageMeter()
            train_surrogate_data_loss = AverageMeter()
            align_loss_avg = AverageMeter()
            aux_acc_avg = AverageMeter()
            proto_loss_avg = AverageMeter()
            correct = 0
            total_num = 0
            agg_noise_protos_label = {}
            for batch_idx, (X_train, _, y_train)in enumerate(dataloader):

                X_train, labels = X_train.to(self.device), y_train.to(self.device)
                real_bs = X_train.size(0)
                aux_data, sampled_label = self.generate_noise_data(
                    noise_label_style=self.args.algorithms.surrogate_label_style)
                aux_data, sampled_label = aux_data.to(self.device), sampled_label.to(self.device)
                
                aux_bs = aux_data.shape[0]
                self.opt.zero_grad()
                self.opt.first_step(g_update)
                x_cat = torch.cat((X_train, aux_data), dim=0)
                feat, outputs = self._forward(inputs=x_cat)

                loss = F.cross_entropy(outputs[0:real_bs], labels)
                ce_loss = loss.item()
                loss_aux = F.cross_entropy(outputs[real_bs:], sampled_label)
                loss += self.args.algorithms.surrogate_alpha * loss_aux
                if self.args.algorithms.surrogate_feat_align:
                    loss_feat_align, align_domain_loss_value, align_cls_loss_value, noise_cls_loss_value = self.proxy_align_loss(
                        feat, torch.cat([labels, sampled_label], dim=0), real_bs)
                    align_loss = loss_feat_align
                    loss +=  align_loss
                if self.args.algorithms.proto_types_align_method == 'mse':
                    if self.global_noise_protos != None:
                        proto_global = torch.zeros_like(feat[real_bs:])
                        for idx, noise_label in enumerate(sampled_label):
                            proto_global[idx,:] = self.global_noise_protos[noise_label.item()]
                        proto_loss = F.mse_loss(proto_global, feat[real_bs:])
                        loss += proto_loss * self.args.algorithms.align_loss_weight
                elif self.args.algorithms.proto_types_align_method == 'triplet':
                    if self.global_noise_protos != None:
                        positive_feat = torch.zeros_like(feat[real_bs:])
                        negative_feat = torch.zeros_like(feat[real_bs:])
                        for idx, noise_label in enumerate(sampled_label):
                            positive_feat[idx,:] = self.global_noise_protos[noise_label.item()]
                            neg_class = (noise_label-self.args.datasets.num_classes + torch.randint(1, self.args.datasets.num_classes, (1,)).item()) % self.args.datasets.num_classes+self.args.datasets.num_classes
                            negative_feat[idx,:] = self.global_noise_protos[neg_class.item()]
                        proto_loss = triplet_loss(feat[real_bs:], positive_feat, negative_feat)
                        loss += proto_loss * self.args.algorithms.align_loss_weight
                self.opt.zero_grad()
                self._backward(loss)
                self.opt.second_step()

                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.args.algorithms.max_norm)
                self.base_opt.step()
                
                # get noise feat for pro
                noise_feat = feat[real_bs:].detach().cpu()
                for i in range(len(sampled_label)):
                    label = sampled_label[i].item()
                    if label in agg_noise_protos_label:
                        agg_noise_protos_label[label].append(noise_feat[i,:])
                    else:
                        agg_noise_protos_label[label] = [noise_feat[i,:]]

                prec1, _, _ = accuracy(outputs[0:real_bs].data, labels)
                prec2, _, _ = accuracy(outputs[real_bs:].data, sampled_label)
                correct_num, total_bs_num = top_one_accuracy(outputs[0:real_bs].data, labels)
                correct += correct_num
                total_num += total_bs_num
                
                train_loss_avg.update(loss.item(), real_bs)
                train_acc_avg.update(prec1, real_bs)
                train_ce_loss_avg.update(ce_loss, real_bs)
                train_surrogate_data_loss.update(loss_aux.item(), aux_bs)
                align_loss_avg.update(align_loss.item(), 1)
                aux_acc_avg.update(prec2, aux_bs)
                if self.global_noise_protos != None:
                    proto_loss_avg.update(proto_loss.item(), aux_bs)    
            acc = 100. * correct / total_num
            self.local_noise_protos, self.proto_agg_weights = local_proto_agg(agg_noise_protos_label)
            
            scalar_dict = {'{role}:{index} old averager acc'.format(role=self.role, index=self.index):train_acc_avg.avg,
                            '{role}:{index} new real acc'.format(role=self.role, index=self.index): acc,
                    '{role}:{index} loss'.format(role=self.role, index=self.index):train_loss_avg.avg,
                    '{role}:{index} ce_loss'.format(role=self.role, index=self.index):train_ce_loss_avg.avg,
                    '{role}:{index} surrogate_data_loss'.format(role=self.role, index=self.index):train_surrogate_data_loss.avg,
                    '{role}:{index} align_loss'.format(role=self.role, index=self.index):align_loss_avg.avg,
                    '{role}:{index} surrogate_data_acc'.format(role=self.role, index=self.index):aux_acc_avg.avg}
            if self.global_noise_protos != None:
                scalar_dict['{role}:{index} proto_loss'.format(role=self.role, index=self.index)] = proto_loss_avg.avg
            if self.args.record:
                record(record_tool=self.args.record_tool, scalar = scalar_dict, step = self.forward_time)

        
        self.model.cpu()
        self.post_train(iteration=train_epoch*len(dataloader))


    def post_train(self,iteration):
        self.model_delta.clear()
        self.delta_c.clear()
        self.grad_pert_corr_delta.clear()
        with torch.no_grad():
            if self.args.share_with_nongrad:
                for name, param in self.model.state_dict().items():
                    self.model_delta[name] = param.data - self.global_model_params[name].data
            else:
                for name, param in self.model.named_parameters():
                    self.model_delta[name] = param.data - self.global_model_params[name].data
        

    def upload_info(self):
        if self.args.algorithms.weighted_sum:
            upload_info = {"ROLE": self.role,
                        "LOCAL_MODEL_DELTA_PARAM": copy.deepcopy(self.model_delta),
                        "SAMPLE_NUM": self.sample_num}
        else:
            upload_info = {"ROLE": self.role,
                        "LOCAL_MODEL_DELTA_PARAM": copy.deepcopy(self.model_delta),
                        "SAMPLE_NUM": 1.0,
                        "LOCAL_NOISE_PROTOS_PARAM": self.local_noise_protos,
                        "PROTO_AGG_WEIGHT": self.proto_agg_weights}

        upload_info['UPLOAD_AMOUNT'] = self.upload_info_account(upload_info)
        return upload_info

    def local_train(self, round_idx, **kwargs):
        self.train(round_idx, self.args.algorithms.local_epochs, self.local_train_lb_dl, pre_flag=kwargs['pre_flag'])


    def compute_noise_proto_distance(self, prototype_dict):
        iterations = 5
        self.model.to(self.device)
        self.model.eval()
        total_distance = 0
        if prototype_dict != None:
            with torch.no_grad():
                for i in range(iterations):
                    aux_data, sampled_label = self.generate_noise_data(
                        noise_label_style=self.args.algorithms.surrogate_label_style)
                    aux_data, sampled_label = aux_data.to(self.device), sampled_label.to(self.device)
                    feat, _ = self._forward(inputs=aux_data)
                    noise_feat = feat.detach().cpu()

                    # compute the distance between feat and proto_global
                    proto_global = torch.zeros_like(noise_feat)
                    for idx, noise_label in enumerate(sampled_label):
                        proto_global[idx,:] = prototype_dict[noise_label.item()]
                    distance = torch.norm(noise_feat - proto_global, dim=1)
                    total_distance += distance.mean()
            avg_distances = total_distance / iterations  
            if self.args.record:
                record(record_tool=self.args.record_tool, scalar = {'{role}:{index} noise proto distance'.format(role=self.role, index=self.index): avg_distances}, step = self.forward_time)
            return avg_distances
        else:
            return None


def local_proto_agg(protos):
    """
    Returns the average of the weights.
    """
    new_protos = {}
    proto_agg_weights = {}
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            new_protos[label] = proto / len(proto_list)
            proto_agg_weights[label] = len(proto_list)
        else:
            new_protos[label] = proto_list[0]
            proto_agg_weights[label] = len(proto_list[0])

    return new_protos, proto_agg_weights


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        
        # Triplet Loss
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
