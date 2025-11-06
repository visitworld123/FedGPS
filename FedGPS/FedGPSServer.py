import copy
import logging
import warnings
from collections import OrderedDict
from tqdm import tqdm

import torch

from  baseFedAvg import strategies
from utils.averager import AverageMeter
from utils.metric import (
    accuracy,
    top_one_accuracy
)
from utils.utils import compare_state_dicts
from utils.utils import zero_weights
from utils.record import record
from utils.utils import check_remove_batch_norm_stats_info
from baseFedAvg.server import Server
from FedGPS.utils import proto_aggregation
class FedGPSServer(Server):
    def __init__(self,  
                 args, 
                 device, 
                 index, 
                 X_train, 
                 y_train, 
                 X_test, 
                 y_test) -> None:
        super().__init__(
                 args, 
                 device, 
                 index, 
                 X_train, 
                 y_train, 
                 X_test, 
                 y_test
        )
        self.global_control_model = copy.deepcopy(self.model)
        zero_weights(self.global_control_model)
        self.global_grad_pert_corr_model = copy.deepcopy(self.model)
        zero_weights(self.global_grad_pert_corr_model)
    def _initial_setup(self):
        if self.args.algorithms.surrogate:
            if self.args.algorithms.surrogate_label_shift:
                self.model = self._build_model(num_classes=self.args.datasets.num_classes+
                                               self.args.datasets.num_classes)
            else:
                self.model = self._build_model()
        self._build_test_dl()
        
    def aggregation(self, agg='general'):

        data_num_weighted = self._reorganize_uploded_info("SAMPLE_NUM")
        clients_model_deltas = self._reorganize_uploded_info("LOCAL_MODEL_DELTA_PARAM")
        clients_noise_protos = self._reorganize_uploded_info("LOCAL_NOISE_PROTOS_PARAM")
        clients_proto_agg_weights = self._reorganize_uploded_info("PROTO_AGG_WEIGHT")
        self.agg_noise_protos = proto_aggregation(clients_noise_protos, clients_proto_agg_weights)
        
        if len(self.clients_uploaded_message) == 0:
            warnings.warn("At aggregation stage, there is no client model") 
        else:  
            #  Update global model
            logging.info("updata global model by the number of data weighted")
            agg_model_delta = strategies.federated_averaging_by_params(clients_model_deltas, data_num_weighted)
            self.global_update = copy.deepcopy(agg_model_delta)
            if not hasattr(self, 'global_ema_update'):
                self.global_ema_update = copy.deepcopy(self.global_update)
            else:
                for name, param in self.global_ema_update.items():
                    param.data.copy_(param.data * self.args.algorithms.ema_decay + self.global_update[name].data * (1 - self.args.algorithms.ema_decay))
            if self.args.share_with_nongrad:
                new_model_params = copy.deepcopy(self.model.state_dict())
                for name, param in agg_model_delta.items():
                    if new_model_params[name].dtype != param.data.dtype:
                        new_model_params[name] = new_model_params[name].to(param.data.dtype)
                    new_model_params[name] += param.data
                self.model.load_state_dict(new_model_params)
            else:
                for name, param in self.model.named_parameters():
                    param.data.copy_(param.data + agg_model_delta[name].data)

            
        self.clients_uploaded_message.clear()


    def download_info(self, round):
        down_info = dict()
        down_info['GLOBAL_MODEL_PARAM'] = self.get_params(self.model, 
                                                            type='model', 
                                                            with_nograd=self.args.share_with_nongrad)
        if round == 0:
            down_info['GLOBAL_UPDATE_PARAM'] = None
            down_info['GLOBAL_EMA_UPDATE_PARAM'] = None
            down_info['GLOBAL_NOISE_PROTOS_PARAM'] = None
        else:
            down_info['GLOBAL_UPDATE_PARAM'] = self.global_update
            down_info['GLOBAL_EMA_UPDATE_PARAM'] = self.global_ema_update
            down_info['GLOBAL_NOISE_PROTOS_PARAM'] = self.agg_noise_protos
        return down_info

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
