import torch
import logging
import copy
from collections import defaultdict
from torch.optim import Optimizer


def param_to_vector(model, type='model'):
    vec = []
    if type == 'model':
        for param in model.parameters():
            vec.append(copy.deepcopy(param).reshape(-1))
    elif type == 'dict':
        for key,param in model.items():
            if 'running' in key or 'num_batches_tracked' in key:
                continue
            vec.append(copy.deepcopy(param).reshape(-1))
    return torch.cat(vec)

class FedGPSOptimizer(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FedGPSOptimizer, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
        self.paras = None
        

    @torch.no_grad()
    def first_step(self,g_update):
        grad_norm = 0
        for group in self.param_groups:
            for idx,p in enumerate(group["params"]):
                p.requires_grad = True 
                if g_update ==None: 
                    break
                else:
                    grad_norm-=g_update[idx].norm(p=2)
                

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx,p in enumerate(group["params"]):
                p.requires_grad = True 
                if g_update ==None: 
                    break
                else:
                    e_w=-g_update[idx] * scale.to(p)
                p.add_(e_w * 1)  
                self.state[p]["e_w"] = e_w
                

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])  
                self.state[p]["e_w"] = 0
    def step(self,g_update):

        inputs, labels, loss_func, model,delta_list,lamb = self.paras

        self.zero_grad()

        self.first_step(g_update)

        param_list = param_to_vector(model)
        predictions = model(inputs)
        loss = loss_func(predictions, labels,param_list,delta_list,lamb)
        self.zero_grad()
        loss.backward()

        self.second_step()
