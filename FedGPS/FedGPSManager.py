import logging

from typing import Dict
from tqdm import tqdm

from baseFedAvg.manager import FedManager
from FedGPS.FedGPSServer import FedGPSServer
from FedGPS.FedGPSClient import FedGPSClient
from utils.utils import set_random
from utils.record import record
class FedGPSManager(FedManager):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.lr_control = self.args.optim.lr
        self.pre_selected_clients = []
    def _setup_server(self):
        self.server = FedGPSServer(args=self.args,
                             device=self.server_device_dict['server'],
                             index=1,
                             X_train=None,
                             y_train=None,
                             X_test=self.X_test,
                             y_test=self.y_test_np)
        
    def _setup_clients(self):
        self.clients_dict: Dict[int:FedGPSClient] = {}
        for i in self.client_X_train_dict.keys():
            self.clients_dict[i] = FedGPSClient(args=self.args,
                                          device=self.client_device_dict[i],
                                          index=i,
                                          X_train=self.client_X_train_dict[i],
                                          y_train=self.client_y_train_dict[i],
                                          X_test=self.X_test,
                                          y_test=self.y_test_np)
            
    def train(self):
        set_random(self.args.train_seed)
        for round in tqdm(range(self.args.global_round), desc='Global Communication Round'):
            
            download_info = self.server.download_info(round)
            client_indexes = self.client_sampling(    # 每一轮Sample一些client
                self.args.client_number,
                self.args.client_number_per_round)
            logging.info("This Round {} the sampled clients is {} ".format(round, client_indexes))
            self.train_locally_per_round(round, client_indexes, download_info)
            self.server.aggregation()

            if round % 5 == 0 and round != 0:
                total_noise_proto_distance = 0
                for client_index in self.clients_dict.keys():
                    client_noise_proto_distance = self.clients_dict[client_index].compute_noise_proto_distance(download_info['GLOBAL_NOISE_PROTOS_PARAM'])
                    total_noise_proto_distance += client_noise_proto_distance
                avg_noise_proto_distance = total_noise_proto_distance / len(client_indexes)
                logging.info("This Round {} the average noise proto distance is {}".format(round, avg_noise_proto_distance))
                scalar_dict = {'global server avg noise proto distance'.format(role='manager', index=1): avg_noise_proto_distance}
                if self.args.record:
                    record(record_tool=self.args.record_tool, scalar = scalar_dict, step = round)
                    
            self.server.test_server(round)
            self.pre_selected_clients = client_indexes

    def train_locally_per_round(self, round, selected_clients, download_info):
        upload_info = dict()
        for num, client_index in enumerate(selected_clients):
            if client_index in self.pre_selected_clients:
                logging.info("This Round {} the client {} in the previous round{}".format(round, client_index, self.pre_selected_clients))
                pre_flag = True
            else:
                logging.info("This Round {} the client {} \"\"not\"\" in the previous round{}".format(round, client_index, self.pre_selected_clients))
                pre_flag = False
            # selected clients parallel running the following steps.
            client: FedGPSClient = self.clients_dict[client_index]
            client.client_update_local_info(download_info)

            client.local_train(round, 
                               pre_flag=pre_flag)
            
            upload_info[client.get_index()] = client.upload_info()
        self.server.receive_message(upload_info)
        logging.info("######################### sampling client_indexes = %s finished the update and upload the info" % str(selected_clients))
