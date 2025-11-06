import copy
import logging
import math

import numpy as np
import torch
from tqdm import tqdm
import cv2


from model.ResNet import ResNet18, ResNet50
from utils.averager import AverageMeter
from utils.metric import (
    accuracy,
    top_one_accuracy
)
from data_preprocessing.personalized_dataset import (
    Dataset_Normal,
    Dataset_WeakStrong
)
from data_preprocessing.utils import record_y_distribution
from data_preprocessing.own_transforms import get_stransform
from utils.record import record
from utils.imp_utils import (
    noisification,
    noisification_in_client
)


class Client(object):

    def __init__(self, 
                 args,
                 device,
                 index,
                 X_train, 
                 y_train, 
                 X_test, 
                 y_test,
                 **kwargs) -> None:
        
        self.args = args
        self.device = device
        self.index = index

        self.X_train = X_train
        self.y_true_train = y_train
        self.y_train = copy.deepcopy(self.y_true_train)
        self.X_test = X_test
        self.y_test = y_test
        
        self.role = 'client'
        self.sample_num = len(self.y_true_train)
        self._local_data_stats()
        self._initial_setup()
        self.forward_time = 0
        self.backward_time = 0
        self._build_data_loader()
        if 'random_noise_level' in kwargs:
            self.random_noise_level = kwargs['random_noise_level']
    

    def _initial_setup(self):
        self.model = self._build_model()
        self.opt = self._build_optimizer(self.model.parameters())

        



    def _build_data_loader(self):
        train_trans = get_stransform(self.args.datasets.dataset_name, train=True)
        if 'nl' in self.args.imperfect_scenario.type and 'pl' not in self.args.imperfect_scenario.type:
            self.local_train_lb_ds = Dataset_WeakStrong(data=self.X_train,
                                                        targets=self.y_train,
                                                        ulb=False,
                                                        dataset=self.args.datasets.dataset_name,
                                                        transform=train_trans)
            self.local_train_lb_dl = torch.utils.data.DataLoader(dataset=self.local_train_lb_ds,
                                                                batch_size=self.args.bs, shuffle=True,
                                                                drop_last=False)
        elif 'pl' in self.args.imperfect_scenario.type:
            if self.lb_num > 0:
                self.local_train_lb_ds = Dataset_WeakStrong(data=self.X_train_lb,
                                                            targets=self.y_train_lb,
                                                            ulb=False,
                                                            dataset=self.args.datasets.dataset_name,
                                                            transform=train_trans)
                self.local_train_lb_dl = torch.utils.data.DataLoader(dataset=self.local_train_lb_ds,
                                                                    batch_size=self.args.bs, shuffle=True,
                                                                    drop_last=False)
            if self.ulb_num > 0:
                self.local_train_ulb_ds = Dataset_WeakStrong(data=self.X_train_ulb,
                                                            targets=self.y_train_ulb,
                                                            ulb=True,
                                                            dataset=self.args.datasets.dataset_name,
                                                            transform=train_trans)
                self.local_train_ulb_dl = torch.utils.data.DataLoader(dataset=self.local_train_ulb_ds,
                                                                 batch_size=self.args.bs, shuffle=True,
                                                                 drop_last=False)
        elif 'general' in self.args.imperfect_scenario.type:
            self.local_train_lb_ds = Dataset_WeakStrong(data=self.X_train,
                                            targets=self.y_true_train,
                                            ulb=False,
                                            dataset=self.args.datasets.dataset_name,
                                            transform=train_trans)
            self.local_train_lb_dl = torch.utils.data.DataLoader(dataset=self.local_train_lb_ds,
                                                                 batch_size=self.args.bs, shuffle=True,
                                                                 drop_last=False)
        
    def _forward(self, inputs=None, mode='general',**kwargs):
        feat, outputs = self.model(inputs)
        if mode == 'general':
            self.forward_time += 1
        elif mode == 'test':
            pass
        else:
            raise NotImplementedError
        return feat, outputs

    def _backward(self, loss=None, mode='general', **kwargs):
        self.backward_time += 1
        loss.backward()
        return 
            
    def get_data_distribution(self):
        return record_y_distribution(self.y_true_train)
    
    def get_model_params(self):
        return {k: copy.deepcopy(val.cpu())
                for k, val in self.model.state_dict().items()}
    
    def get_model_params_wo_stat_info(self):
        return [copy.deepcopy(val.cpu())
                for val in self.model.parameters()]
    
    def _local_data_stats(self):

        self.cls_map = dict()
        self.cls_arr, self.cls_cnt_arr = np.unique(self.y_true_train, return_counts=True)
        for i in range(len(self.cls_arr)):
            self.cls_map[self.cls_arr[i]] = np.where(self.y_true_train == self.cls_arr[i])[0]
        

    def get_labeled_data_distribution(self):
        if self.lb_num > 0:
            return record_y_distribution(self.y_train_lb)
        else:
            return None

    def set_model_params(self, params, with_nongrad=True):
        self.model.load_state_dict(params)
    
    def set_model_params_wo_stat_info(self, params):
        for idx, val in enumerate(self.model.parameters()):
            val.data.copy_(params[idx])
    
    def client_update_local_info(self, download_info):

        if 'GLOBAL_MODEL_PARAM' in download_info:
            logging.info("Client {} update local model with global model".format(self.index))
    
            # FedInit use Relaxed Initialization
            if self.args.algorithms.algorithm_name == 'FedInit':
                logging.info("Use Relaxed Initialization")
                model_params = copy.deepcopy(download_info['GLOBAL_MODEL_PARAM'])

                if self.args.share_with_nongrad:
                    local_params = copy.deepcopy(self.get_model_params())
                    for name in model_params.keys():
                        model_params[name] = model_params[name] + self.args.algorithms.beta * (model_params[name] - local_params[name])
                    self.set_model_params(model_params)
                else:
                    local_params = copy.deepcopy(self.get_model_params_wo_stat_info())
                    for idx in range(len(model_params)):
                        model_params[idx] = model_params[idx] + self.args.algorithms.beta * (model_params[idx] - local_params[idx])
                    self.set_model_params_wo_stat_info(model_params)
            else:
                # usual FedAvg
                if self.args.share_with_nongrad:
                    self.set_model_params(copy.deepcopy(download_info['GLOBAL_MODEL_PARAM']), with_nongrad=self.args.share_with_nongrad)
                else:
                    self.set_model_params_wo_stat_info(copy.deepcopy(download_info['GLOBAL_MODEL_PARAM']))
            
            self.global_model_params = copy.deepcopy(download_info['GLOBAL_MODEL_PARAM'])
        else:
            raise ValueError('Client does not receive global model')

    def _build_model(self, model_name=None, num_classes=None, input_channels=None):
        if model_name is not None:
            model = model_name
        else:
            model = self.args.model
        if num_classes is None:
            num_classes = self.args.datasets.num_classes
        if input_channels is None:
            input_channels = self.args.datasets.input_channels
        else:
            input_channels = input_channels

        if model == 'resnet18':
            net = ResNet18(self.args, num_classes, input_channels)
        elif model == 'resnet50':
            net = ResNet50(self.args, num_classes, input_channels)
        else:
            NotImplementedError
        return net
    
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
    
        if self.args.optim.optim_name == 'SGD':
            if momentum < 1.0:
                optimizer = torch.optim.SGD(params_to_optimizer,
                        lr=lr,
                        weight_decay=weight_decay,
                        momentum=momentum,
                        nesterov=self.args.optim.nesterov)
            else:
                optimizer = torch.optim.SGD(params_to_optimizer,
                                            lr=lr,
                                            weight_decay=weight_decay,
                                            nesterov=self.args.optim.nesterov)

        elif self.args.algorithms.algorithm_name == 'FedProx':
            pass
        else:
            NotImplementedError
    
        return optimizer

    def test(self,
             round_idx,
             testloader):
        self.model.eval()
        self.model.to(self.device)

        loss_avg = AverageMeter()
        acc_avg = AverageMeter()
        total_num = 0
        correct = 0

        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                bs = inputs.size(0)

                _, outputs = self._forward(inputs=inputs, mode='test')
                loss = criterion(outputs, targets)
                prec1, _, _ = accuracy(outputs.data, targets)
                correct_num, total_bs_num = top_one_accuracy(outputs.data, targets)

                correct += correct_num
                total_num += total_bs_num
                loss_avg.update(loss.item(), bs)
                acc_avg.update(prec1, bs)
        acc = 100. * correct / total_num

        scalar_dict = {'{role}:{index} old averager acc'.format(role=self.role, index=self.index):acc_avg.avg,
                       '{role}:{index} new real acc'.format(role=self.role, index=self.index):acc,
                       '{role}:{index} loss'.format(role=self.role, index=self.index):loss_avg.avg,
                       'Test Round':round_idx}
        if self.args.record:
            record(record_tool=self.args.record_tool,scalar = scalar_dict, step = round_idx)

        return acc


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
    
    def train(self,
              round,
              train_epoch,
              dataloader,
              **kwargs):
        
        logging.info("-------------------local train-------------------")      
        self.model.to(self.device)
        self.model.train()
        
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(train_epoch):
            logging.info('=> Training Epoch #%d, LR=%.4f' % (epoch, self.opt.param_groups[0]['lr']))
            model_before_train = copy.deepcopy(self.model.state_dict())
            train_acc_avg = AverageMeter()
            train_loss_avg = AverageMeter()


            
            correct = 0
            total_num = 0
            if self.args.algorithms.algorithm_name == 'FedProx':
                prox_loss_avg = AverageMeter()
            if self.args.algorithms.algorithm_name == 'FedDecorr':
                feddecorr_loss_avg = AverageMeter()
            for batch_idx, (X_train, _, y_train)in enumerate(dataloader):

                X_train, labels = X_train.to(self.device), y_train.to(self.device)
                bs = X_train.size(0)
                self.opt.zero_grad()
                
                feats, outputs = self._forward(inputs=X_train)

                loss = criterion(outputs, labels)
                
              
                self._backward(loss=loss)
                if self.args.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.args.max_norm)

                self.opt.step()

                prec1, _, _ = accuracy(outputs.data, labels)
                correct_num, total_bs_num = top_one_accuracy(outputs.data, labels)
                correct += correct_num
                total_num += total_bs_num
                
                train_loss_avg.update(loss.item(), bs)
                train_acc_avg.update(prec1, bs)
                

            model_after_train = copy.deepcopy(self.model.state_dict())
            local_update_direction = self.calculate_local_update_direction(model_before_train, model_after_train)
            linear_local_update_dir = local_update_direction['linear.weight']
            conv_local_update_dir = local_update_direction['conv1.weight']
        
            acc = 100. * correct / total_num

            scalar_dict = {'{role}:{index} old averager acc'.format(role=self.role, index=self.index):train_acc_avg.avg,
                            '{role}:{index} new real acc'.format(role=self.role, index=self.index): acc,
                    '{role}:{index} loss'.format(role=self.role, index=self.index):train_loss_avg.avg}
           
            if self.args.record:
                record(record_tool=self.args.record_tool, scalar = scalar_dict, step = self.forward_time)

        self.model.cpu()


    def local_train(self, round_idx, **kwargs):
        self.train(round_idx, self.args.algorithms.local_epochs, self.local_train_lb_dl)
    
    def train_using_lb_data(self, round_idx, **kwargs):
        self.train(round_idx, self.args.algorithms.local_epochs, self.local_train_lb_dl)

    def has_labeled_data(self):
        return True if self.lb_num > 0 else False
            
    def upload_info_account(self, upload_info):
        # compute the communication cost 
        total_params = 0
        for info_key in upload_info.keys():
            if 'PARAM' in info_key:
                total_params += sum(upload_info[info_key][key].numel() for key in upload_info[info_key].keys())
        logging.info("The total information size is {} sent to server of client {}".format(total_params, self.index))
        return total_params
    
    
    def upload_info(self):
        # compute the communication cost 
        upload_info = {"ROLE": self.role,
                       "SAMPLE_NUM": self.sample_num}
        if self.args.share_with_nongrad:
            upload_info["LOCAL_MODEL_PARAM"] = self.get_model_params()
        else:
            upload_info["LOCAL_MODEL_PARAM"] = self.get_model_params_wo_stat_info()

        upload_info['UPLOAD_AMOUNT'] = self.upload_info_account(upload_info)
        return upload_info
    
    def get_index(self):
        return self.index
    

    def update_lr(self, lr):
        self.opt.param_groups[0]['lr'] = lr

    def labeded_data_training(self,round,train_epoch,**kwargs):
        logging.info("-------------------trainining only use labeled data-------------------")      
        self.model.to(self.device)
        self.model.train()
        
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(train_epoch):
            logging.info('=> Training Epoch #%d, LR=%.4f' % (epoch, self.opt.param_groups[0]['lr']))
            train_acc_avg = AverageMeter()
            train_loss_avg = AverageMeter()
            correct = 0
            total_num = 0
            for batch_idx, (X_train, _, y_train)in enumerate(self.local_train_lb_dl):

                X_train, labels = X_train.to(self.device), y_train.to(self.device)
                bs = X_train.size(0)
                self.opt.zero_grad()
                
                feats, outputs = self._forward(inputs=X_train)

                loss = criterion(outputs, labels)
                
                self._backward(loss=loss)

                if self.args.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.args.max_norm)

                self.opt.step()

                prec1, _, _ = accuracy(outputs.data, labels)
                correct_num, total_bs_num = top_one_accuracy(outputs.data, labels)
                correct += correct_num
                total_num += total_bs_num
                
                train_loss_avg.update(loss.item(), bs)
                train_acc_avg.update(prec1, bs)
            acc = 100. * correct / total_num
            
            scalar_dict = {'{role}:{index} old averager acc'.format(role=self.role, index=self.index):train_acc_avg.avg,
                            '{role}:{index} new real acc'.format(role=self.role, index=self.index): acc,
                    '{role}:{index} loss'.format(role=self.role, index=self.index):train_loss_avg.avg}
            if self.args.record:
                record(record_tool=self.args.record_tool, scalar = scalar_dict, step = self.forward_time)

        self.model.cpu()

    def calculate_pert_grad(self, pert_grad, grad_after_pert):
        pert_grad = pert_grad.to(self.device).view(-1)
        grad_after_pert = grad_after_pert.to(self.device).view(-1)
        cos_sim = torch.nn.functional.cosine_similarity(pert_grad, grad_after_pert, dim=0)
        angle = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180 / torch.pi
        gv = grad_after_pert - (torch.norm(grad_after_pert) * cos_sim * pert_grad / torch.norm(pert_grad))


        return cos_sim, angle, torch.norm(gv)
    
    def calculate_local_update_direction(self, model_before_train, model_after_train):
        local_update_direction = {}
        for key in model_before_train.keys():
            local_update_direction[key] = model_after_train[key] - model_before_train[key]
        return local_update_direction