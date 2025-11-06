import copy
import logging
import warnings
from collections import OrderedDict
import numpy as np
import torch
import os

from utils.averager import AverageMeter
from data_preprocessing.own_transforms import get_stransform
from data_preprocessing.personalized_dataset import Dataset_Normal
from  baseFedAvg import strategies
from utils.metric import (
    accuracy,
    top_one_accuracy
)
from model.ResNet import ResNet18, ResNet50
from utils.record import record

class Server(object):
    def __init__(self, 
                 args, 
                 device, 
                 index, 
                 X_train, 
                 y_train, 
                 X_test, 
                 y_test) -> None:
        
        self.role='server'

        self.args = args
        self.device = device
        self.index = index

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self._initial_setup()
        self.global_acc = 0.0
        self.best_model = None

    def _initial_setup(self):
        self.model = self._build_model()
        self._build_test_dl()

    def _build_model(self, model_name=None, num_classes=None, input_channels=None):
        if model_name is not None:
            model = model_name
        else:
            model = self.args.model
        if num_classes is None:
            num_classes = self.args.datasets.num_classes
        if input_channels is None:
            input_channels = self.args.datasets.input_channels

        if model == 'resnet18':
            net = ResNet18(self.args, num_classes, input_channels)
        elif model == 'resnet50':
            net = ResNet50(self.args, num_classes, input_channels)
        else:
            NotImplementedError
        return net
    
    def _forward(self, inputs):
        feat, outputs = self.model(inputs)
        return feat, outputs
    
    def _set_test_mode(self):
        self.model.eval()
        self.model.to(self.device)

    def test(self,
             round_idx,
             testloader):
        
        self._set_test_mode()

        loss_avg = AverageMeter()
        acc_avg = AverageMeter()
        total_num = 0
        correct = 0

        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                bs = inputs.size(0)

                _, outputs = self._forward(inputs)
                loss = criterion(outputs, targets)
                prec1, _, _ = accuracy(outputs.data, targets)
                correct_num, total_bs_num = top_one_accuracy(outputs.data, targets)

                correct += correct_num
                total_num += total_bs_num
                loss_avg.update(loss.item(), bs)
                acc_avg.update(prec1, bs)
        acc = 100. * correct / total_num

        scalar_dict = {'Test {role}:{index} old averager acc'.format(role=self.role, index=self.index):acc_avg.avg,
                       'Test {role}:{index} new real acc'.format(role=self.role, index=self.index):acc,
                       'Test {role}:{index} loss'.format(role=self.role, index=self.index):loss_avg.avg,
                       'Test Round':round_idx}
        if self.args.record:
            record(record_tool=self.args.record_tool, scalar = scalar_dict, step = round_idx)
        if acc > self.global_acc:
            logging.info(f"This round: {round_idx}, Global acc is: {acc}, new best acc is: {acc}")
            self.global_acc = acc
            self.best_model = self.get_model_params()
        self.model.cpu()
        return acc
    
    def _build_test_dl(self):
        test_trans = get_stransform(self.args.datasets.dataset_name, train=False)

        test_ds = Dataset_Normal(data=self.X_test,
                                 targets=self.y_test,
                                 dataset=self.args.datasets.dataset_name,
                                 transform=test_trans)
        self.test_dl = torch.utils.data.DataLoader(dataset=test_ds,
                                                   batch_size=self.args.bs, shuffle=True,
                                                   drop_last=False)

    def test_server(self, round):
        acc = self.test(round, self.test_dl)
        logging.info(f"This round: {round}, Global acc is: {acc}")
        return acc

    def download_info(self):
        if self.args.share_with_nongrad:
            down_info = {"GLOBAL_MODEL_PARAM": copy.deepcopy(self.get_model_params())}
        else:
            down_info = {"GLOBAL_MODEL_PARAM": copy.deepcopy(self.get_model_params_wo_stat_info())}
        if hasattr(self, 'model_differences'):
            down_info["GLOABL_UPDATE_DIRECTION"] = copy.deepcopy(self.model_differences)
        return down_info
    
    def receive_message(self, upload_message):

        self.clients_uploaded_message = upload_message
        """
        a dict of clients info 
        {index: 
                {"ROLE": server/client, 
                "LOCAL_MODEL_PARAM": {key:params},
                "SAMPLE_NUM": client_num}, 
            , ... }
        """
    def _reorganize_uploded_info(self, target_key):
        key_uploaded_info = dict()
        # key is the index of client
        for key in self.clients_uploaded_message.keys():
            if target_key in self.clients_uploaded_message[key]:
                key_uploaded_info[key] = self.clients_uploaded_message[key][target_key]
        
        return key_uploaded_info
    
  
    def calculate_client_model_differences(self, clients_params):
        """Calculate the difference between each client's model parameters and the server model parameters.
        
        Args:
            clients_params (dict): Dictionary of client parameters {client_idx: model_params}
        
        Returns:
            dict: Dictionary containing differences for each client {client_idx: {param_name: difference}}
        """
        total_differences = {}
        stat_diff = {}
        server_params = self.get_model_params()
        for client_idx, client_params in clients_params.items():
            client_diff = {}
            scalar_dict = {}
            total_differences[client_idx] = 0
            stat_diff[client_idx] = 0
            for param_name, server_param in server_params.items():
                if param_name in client_params:
                    # Calculate difference using L2 norm
                    diff = torch.norm(client_params[param_name].float() - server_param.float()).item()
                    total_differences[client_idx] += diff
                    if 'running' in param_name:
                        stat_diff[client_idx] += diff
                    scalar_dict[f'{client_idx}_diff_{param_name}'] = diff
            scalar_dict[f'{client_idx}_stat_diff'] = stat_diff[client_idx]
            scalar_dict[f'{client_idx}_total_diff'] = total_differences[client_idx]
            if self.args.record:
                record(record_tool=self.args.record_tool,scalar = scalar_dict, step = 1)            
            logging.info(f'Client {client_idx} total difference: {total_differences[client_idx]}')
            logging.info(f'Client {client_idx} stat difference: {stat_diff[client_idx]}')


    def aggregation(self, agg='general'):
        

        data_num_weighted = self._reorganize_uploded_info("SAMPLE_NUM")
        clients_params = self._reorganize_uploded_info("LOCAL_MODEL_PARAM")
        self.calculate_client_model_differences(copy.deepcopy(clients_params))
        if len(self.clients_uploaded_message) == 0:
            warnings.warn("At aggregation stage, there is no client model") 
        else:   
            logging.info("updata global model by the number of data weighted")
            if agg == 'median':
                model_params = strategies.federated_median_by_params(clients_params)
            elif agg == 'general' :
                if self.args.share_with_nongrad:
                    model_params = strategies.federated_averaging_by_params(clients_params, data_num_weighted)
                    if model_params is None:
                        logging.info("No client model to aggregate")
                        return
                else:   
                    model_params = strategies.federated_averaging_by_params_wo_stat_info(clients_params, data_num_weighted)
                    if model_params is None:
                        logging.info("No client model to aggregate")
                        return
            else:
                raise NotImplementedError
            self.model_differences = self.calculate_model_update_direction(model_params)
            if self.args.share_with_nongrad:
                self.set_model_params(model_params)
            else:
                self.set_model_params_wo_stat_info(model_params)
        self.clients_uploaded_message.clear()

    def aggregation_momentum(self, round):
        data_num_weighted = self._reorganize_uploded_info("SAMPLE_NUM")
        clients_params = self._reorganize_uploded_info("LOCAL_MODEL_PARAM")
        self.calculate_client_model_differences(copy.deepcopy(clients_params))
        if len(self.clients_uploaded_message) == 0:
            warnings.warn("At aggregation stage, there is no client model") 
        else:   
            logging.info("updata global model by the number of data weighted with momentum")
            model_params = strategies.federated_averaging_by_params(clients_params, data_num_weighted)
            self.prev_params = copy.deepcopy(self.model.cpu().state_dict())
            pseudo_params = OrderedDict()
            for key in self.prev_params.keys():
                x = self.prev_params[key]
                y = model_params[key]
                pseudo_params[key] = x - y
            if round > 1:
                assert self.momentum_vector, "Momentum should have been created on round 1."
                for key in pseudo_params.keys():
                    self.momentum_vector[key] = self.args.algorithms.fedavgm_momentum * self.momentum_vector[key] + pseudo_params[key]
            else:
                self.momentum_vector = pseudo_params
                
            pseudo_gradient = OrderedDict()
            for key in self.momentum_vector.keys():
                pseudo_gradient[key] = pseudo_params[key] + self.args.algorithms.fedavgm_momentum * self.momentum_vector[key]
            fedavgm_result = OrderedDict()
            for key in pseudo_gradient.keys():
                fedavgm_result[key] = self.prev_params[key] - pseudo_gradient[key]
            if self.args.share_with_nongrad:
                self.set_model_params(model_params)
            else:
                self.set_model_params_wo_stat_info(model_params)

    def get_model_params(self):
        return {k: copy.deepcopy(val.cpu())
                for k, val in self.model.state_dict().items()}
    
    def get_model_params_wo_stat_info(self):
        return [copy.deepcopy(val.cpu())
                for val in self.model.parameters()]
    
    def set_model_params(self, params):
        self.model.load_state_dict(params)
        self.model.cpu()
    
    def set_model_params_wo_stat_info(self, params):
        for idx, val in enumerate(self.model.parameters()):
            val.data.copy_(params[idx])
    
    def calculate_model_update_direction(self, model_params):
        server_params = self.get_model_params()
        model_update_direction = OrderedDict()
        for key in server_params.keys():
            model_update_direction[key] = server_params[key] - model_params[key]
        return model_update_direction
    
