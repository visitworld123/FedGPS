import argparse
import logging
import os
import random
import sys
from omegaconf import ListConfig

import hydra
import numpy as np
import setproctitle
import torch
from omegaconf import DictConfig, OmegaConf
from utils.utils import set_random
from utils.record import *




@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setproctitle.setproctitle("Test")
    torch.backends.cudnn.deterministic = True
    print(OmegaConf.to_yaml(cfg))
  
    logging.info("process ID = " + str(os.getpid()))
    cfg.process_PID = os.getpid()
    algorithm_name = cfg.algorithms.algorithm_name
    if cfg.record and cfg.record_tool == 'wandb':
        import wandb
        os.environ['WANDB_MODE'] = 'online'
        """
        {algorithms}_{comm_round}_{local_epochs}_{seed}_{client_number}_{client_number_per_round}_{model}_{bs}_{partition_alpha}_{use_gradient_clipping}_{imperfect_scenario.type}_{noise_type}_{noise_ratio}_{prox_term}_{feddecorr}
        """
        wandb.init(project=cfg.wandb_project,
                name=cfg.extra_info + "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                algorithm_name,
                cfg.datasets.dataset_name,
                cfg.global_round,
                cfg.algorithms.local_epochs,
                cfg.seed,
                cfg.client_number,
                cfg.client_number_per_round,
                cfg.model,
                cfg.bs,
                cfg.partition_alpha,
                cfg.use_gradient_clipping,
                cfg.imperfect_scenario.type,
                cfg.imperfect_scenario.noise_type,
                cfg.imperfect_scenario.noise_inclient,
                cfg.imperfect_scenario.noise_ratio,
                cfg.train_seed,
                cfg.process_PID
        ),
        config=dict(cfg)
        )
    elif cfg.record and cfg.record_tool == 'swanlab':
        import swanlab
        """
        {algorithms}_{comm_round}_{local_epochs}_{seed}_{client_number}_{client_number_per_round}_{model}_{bs}_{partition_alpha}_{use_gradient_clipping}_{imperfect_scenario.type}_{noise_type}_{noise_ratio}_{prox_term}_{feddecorr}
        """
        swanlab.init(project=cfg.wandb_project,
                name=cfg.extra_info + "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                algorithm_name,
                cfg.datasets.dataset_name,
                cfg.global_round,
                cfg.algorithms.local_epochs,
                cfg.seed,
                cfg.client_number,
                cfg.client_number_per_round,
                cfg.model,
                cfg.bs,
                cfg.partition_alpha,
                cfg.use_gradient_clipping,
                cfg.imperfect_scenario.type,
                cfg.imperfect_scenario.noise_type,
                cfg.imperfect_scenario.noise_inclient,
                cfg.imperfect_scenario.noise_ratio,
                cfg.train_seed,
                cfg.process_PID
        ),
        
        config=dict(cfg)
        )

    else:
        os.environ['WANDB_MODE'] = 'dryrun'
    set_random(cfg.seed)
    if cfg.algorithms.algorithm_name == 'FedAvg':
        from baseFedAvg.manager import FedManager
        manager = FedManager(cfg)
    elif cfg.algorithms.algorithm_name == 'BVAT':
        from FedGPS.FedGPSManager import FedGPSManager
        manager = FedGPSManager(cfg)
    else:
        raise ValueError('Invalid algorithm')
    manager.train()
    if cfg.record and cfg.record_tool == 'wandb':
        wandb.finish()
    elif cfg.record and cfg.record_tool == 'swanlab':
        swanlab.finish()



if __name__ == "__main__":
    main()
