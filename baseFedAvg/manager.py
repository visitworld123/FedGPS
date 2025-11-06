import logging
import copy
import math
from typing import Dict

import torch
import numpy as np
from tqdm import tqdm
from utils.utils import set_random
from data_preprocessing.load import (
    split_data_into_clients
)
from baseFedAvg.server import Server
from baseFedAvg.client import Client

class FedManager(object):
    def __init__(self, args) -> None:
        self.args = args
        self._initial_setup()

    def _initial_setup(self):
        self._setup_device()  # {client_idx:gpu_idx}
        self._setup_dataset() 
        self._setup_server()
        if 'nl' in self.args.imperfect_scenario.type and self.args.imperfect_scenario.noise_type == 'random':
            self._random_noise_simulation()
        self._setup_clients()
        
    
    def _random_noise_simulation(self):
        level_n_system, level_n_lowerb = self.args.imperfect_scenario.random_nl_ratio
        gamma_s = np.random.binomial(1, level_n_system, self.args.client_number) # system level noise
        gamma_c_initial = np.random.rand(self.args.client_number)
        gamma_c_initial = (1 - level_n_lowerb) * gamma_c_initial + level_n_lowerb
        self.random_nl_system_level_list = gamma_s * gamma_c_initial
        logging.info(self.random_nl_system_level_list)
    def _setup_device(self):
        self.client_device_dict = {} # {client_idx:gpu_idx}
        device_num = len(self.args.gpu_list)
        for i in range(self.args.client_number):
            self.client_device_dict[i] =  torch.device("cuda:" + str(self.args.gpu_list[i%device_num]) if torch.cuda.is_available() else "cpu")
        self.server_device_dict = {'server':torch.device("cuda:" + str(self.args.gpu_list[0]) if torch.cuda.is_available() else "cpu")} 

    def _setup_dataset(self):
        if self.args.long_tail:
            self.client_X_train_dict, self.client_y_train_dict, self.X_test, self.y_test_np=split_data_into_clients(
                                                                            dataset=self.args.datasets.dataset_name,
                                                                            datadir=self.args.datasets.data_dir,
                                                                            client_number=self.args.client_number,
                                                                            partition_method=self.args.partition_method,
                                                                            long_tail=self.args.long_tail,
                                                                            class_num=self.args.datasets.num_classes,
                                                                            partition_alpha=self.args.partition_alpha,
                                                                            seed=self.args.seed,
                                                                            long_tail_imb_factor=self.args.long_tail_imb_factor
                                                                            )
        else:
            self.client_X_train_dict, self.client_y_train_dict, self.X_test, self.y_test_np=split_data_into_clients(
                                                                                        dataset=self.args.datasets.dataset_name,
                                                                                        datadir=self.args.datasets.data_dir,
                                                                                        client_number=self.args.client_number,
                                                                                        partition_method=self.args.partition_method,
                                                                                        long_tail=self.args.long_tail,
                                                                                        class_num=self.args.datasets.num_classes,
                                                                                        partition_alpha=self.args.partition_alpha,
                                                                                        seed=self.args.seed
                                                                                        )
    def _setup_server(self):
        self.server = Server(args=self.args,
                             device=self.server_device_dict['server'],
                             index=1,
                             X_train=None,
                             y_train=None,
                             X_test=self.X_test,
                             y_test=self.y_test_np)
        
    def _setup_clients(self):
        self.clients_dict: Dict[int:Client] = {}
        for i in self.client_X_train_dict.keys():
            if 'nl' in self.args.imperfect_scenario.type and self.args.imperfect_scenario.noise_type == 'random':
                self.clients_dict[i] = Client(args=self.args,
                                              device=self.client_device_dict[i],
                                              index=i,
                                              X_train=self.client_X_train_dict[i],
                                              y_train=self.client_y_train_dict[i],
                                              X_test=self.X_test,
                                              y_test=self.y_test_np,
                                              random_noise_level=self.random_nl_system_level_list[i])
            else:
                self.clients_dict[i] = Client(args=self.args,
                                            device=self.client_device_dict[i],
                                            index=i,
                                            X_train=self.client_X_train_dict[i],
                                            y_train=self.client_y_train_dict[i],
                                            X_test=self.X_test,
                                            y_test=self.y_test_np)
        
    def update_clients_lr(self, lr, client_indexes):
        for i in client_indexes:
            self.clients_dict[i].update_lr(lr)
            
    def update_optimizer_lr(self, opt, lr):
        opt.param_groups[0]['lr'] = lr

    def client_sampling(self,  client_num_in_total, client_num_per_round):
        # logging.info("Client Sample Probability: %s "%(str(p)))
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            # make sure for each comparison, we are selecting the same clients each round
            num_clients = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.selected_clients = client_indexes

        return client_indexes
    
    def train(self):
        set_random(self.args.train_seed)
        for round in tqdm(range(self.args.global_round), desc='Global Communication Round'):
            
            download_info = self.server.download_info()
            client_indexes = self.client_sampling(    # 每一轮Sample一些client
                self.args.client_number,
                self.args.client_number_per_round)
            logging.info("This Round {} the sampled clients is {} ".format(round, client_indexes))
            self.train_locally_per_round(round, client_indexes, download_info)
            self.server.aggregation()

            self.server.test_server(round)
    
    def train_locally_per_round(self, round, selected_clients, download_info):
        upload_info = dict()
        
        for num, client_index in enumerate(selected_clients):
            # selected clients parallel running the following steps.
            client: Client = self.clients_dict[client_index]
            client.client_update_local_info(download_info)

            client.local_train(round)
            
            upload_info[client.get_index()] = client.upload_info()
        self.server.receive_message(upload_info)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))
            

        
