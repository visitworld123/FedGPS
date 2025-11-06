import copy
from collections import OrderedDict

import torch




def get_average_weight(weights_dict):
    average_weights_dict = dict()
    sum = 0

    for index in weights_dict.keys():
        local_sample_number = weights_dict[index]
        sum += local_sample_number
    if sum == 0:
        return None
    for index in weights_dict.keys():
        local_sample_number = weights_dict[index]

        weight_by_sample_num = local_sample_number / sum

        average_weights_dict[index] = weight_by_sample_num

    return average_weights_dict

def average_named_params_wo_stat_info(named_params_dict, average_weights_dict):
    """
        This is a weighted average operation.
        average_weights_dict_list: includes weights with respect to clients. Same for each param.
        inplace:  Whether change the first client's model inplace.
    """
    # logging.info("################aggregate: %d" % len(named_params_list))
    # averaged_params = copy.deepcopy(named_params_dict.keys()[0])
    with torch.no_grad():
        averaged_params = []
    #   averaged_params is the dict of all parameters      #
        for i, index in enumerate(named_params_dict.keys()): # certain client
            local_param = named_params_dict[index]
            for idx, val in enumerate(local_param):
                w = average_weights_dict[index]
                if i == 0:
                    averaged_params.append(val * w)
                else:
                    averaged_params[idx] += val * w
    return averaged_params

def average_named_params(named_params_dict, average_weights_dict):
    """
        This is a weighted average operation.
        average_weights_dict_list: includes weights with respect to clients. Same for each param.
        inplace:  Whether change the first client's model inplace.
    """
    # logging.info("################aggregate: %d" % len(named_params_list))
    # averaged_params = copy.deepcopy(named_params_dict.keys()[0])
    with torch.no_grad():
        averaged_params = dict()
    #   averaged_params is the dict of all parameters      #
        for i, index in enumerate(named_params_dict.keys()): # certain client
            local_param = named_params_dict[index]
            for k in local_param.keys():
                w = average_weights_dict[index]
                if i == 0:
                    averaged_params[k] = local_param[k] * w
                else:
                    averaged_params[k] += local_param[k] * w
    return OrderedDict(averaged_params)

def average_named_params_norm(named_params_dict, average_weights_dict):
    """
        This is a weighted average operation.
        average_weights_dict_list: includes weights with respect to clients. Same for each param.
        inplace:  Whether change the first client's model inplace.
    """
    # logging.info("################aggregate: %d" % len(named_params_list))
    # averaged_params = copy.deepcopy(named_params_dict.keys()[0])
    params_norm_avg = 0.0
#   averaged_params is the dict of all parameters      #
    for i, index in enumerate(named_params_dict.keys()): # certain client
        local_param = named_params_dict[index]
        for k in local_param.keys():
            w = average_weights_dict[index]
            params_norm_avg += (torch.linalg.norm(local_param[k].to(torch.float32))**2) *w
    return params_norm_avg


def federated_averaging_by_params(models_params, weights):
    """Compute weighted average of model parameters and persistent buffers.
    Using state_dict of model, including persistent buffers like BN stats.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
    """
    average_weights_dict = get_average_weight(weights)
    if average_weights_dict is None:
        return None
    model_params = average_named_params(models_params, average_weights_dict)

    return model_params



def federated_averaging_by_params_wo_stat_info(models_params, weights):
    """Compute weighted average of model parameters and persistent buffers.
    Using state_dict of model, including persistent buffers like BN stats.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
    """
    average_weights_dict = get_average_weight(weights)
    model_params = average_named_params_wo_stat_info(models_params, average_weights_dict)

    return model_params

def federated_averaging_by_params_and_norm(models_params, weights):
    """Compute weighted average of model parameters and persistent buffers.
    Using state_dict of model, including persistent buffers like BN stats.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
    """
    average_weights_dict = get_average_weight(weights)
    model_params = average_named_params(models_params, average_weights_dict)
    params_norm_avg = average_named_params_norm(models_params, average_weights_dict)    

    return model_params, params_norm_avg

def federated_averaging_by_extrapolation():
    model_params = 0
    return model_params



def federated_averaging_by_params_with_additional_weights(models_params, weights, label_flag, flag_weights):
    """Compute weighted average of model parameters and persistent buffers.
    Using state_dict of model, including persistent buffers like BN stats.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
    """
    average_weights_dict_list = get_average_weight(weights)

    averaged_params = copy.deepcopy(models_params[0])
    #   averaged_params is the dict of all parameters      #
    for k in averaged_params.keys():
        for i in range(0, len(models_params)):  # model个数
            if type(models_params[0]) is tuple or type(models_params[0]) is list:
                local_sample_number, local_named_params = models_params[i]
            else:
                local_named_params = models_params[i]
            # logging.debug("aggregating ---- local_sample_number/sum: {}/{}, ".format(
            #     local_sample_number, sum))
            if sum(label_flag) > 0:
                if label_flag[i]:
                    w = average_weights_dict_list[i] * flag_weights  # 不同client的权重
                elif label_flag[i] == False:
                    w = average_weights_dict_list[i] * (1.0-flag_weights)
            elif sum(label_flag) == 0:
                w = average_weights_dict_list[i]
            # w = torch.full_like(local_named_params[k], w).detach()
            if i == 0:
                averaged_params[k] = (local_named_params[k] * w).type(averaged_params[k].dtype)
            else:
                averaged_params[k] += (local_named_params[k].to(averaged_params[k].device) * w).type(
                    averaged_params[k].dtype)

    return averaged_params


def federated_averaging_only_params(models, weights):
    """Compute weighted average of model parameters. Use model parameters only.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
    """
    if models == [] or weights == []:
        return None

    model, total_weights = weighted_sum_only_params(models, weights)
    model_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, params in model_params.items():
            model_params[name].set_(model_params[name] / total_weights)

    return model


def weighted_sum(models, weights):
    """Compute weighted sum of model parameters and persistent buffers.
    Using state_dict of model, including persistent buffers like BN stats.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
        float: Sum of weights.
    """
    if models == [] or weights == []:
        return None
    model = copy.deepcopy(models[0])
    model_sum_params = copy.deepcopy(models[0].state_dict())

    with torch.no_grad():
        for name, params in model_sum_params.items():
            params *= weights[0]
            for i in range(1, len(models)):
                model_params = dict(models[i].state_dict())
                params += model_params[name] * weights[i]
            model_sum_params[name] = params
    model.load_state_dict(model_sum_params)
    return model, sum(weights)


def weighted_sum_only_params(models, weights):
    """Compute weighted sum of model parameters. Use model parameters only.

    Args:s
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
        float: Sum of weights.
    """
    if models == [] or weights == []:
        return None

    model_sum = copy.deepcopy(models[0])
    model_sum_params = dict(model_sum.named_parameters())

    with torch.no_grad():
        for name, params in model_sum_params.items():
            params *= weights[0]
            for i in range(1, len(models)):
                model_params = dict(models[i].named_parameters())
                params += model_params[name] * weights[i]
            model_sum_params[name].set_(params)
    return model_sum


def equal_weight_averaging(models):
    if models == []:
        return None

    model_avg = copy.deepcopy(models[0])
    model_avg_params = dict(model_avg.named_parameters())

    with torch.no_grad():
        for name, params in model_avg_params.items():
            for i in range(1, len(models)):
                model_params = dict(models[i].named_parameters())
                params += model_params[name]
            model_avg_params[name].set_(params / len(models))
    return model_avg

def median_named_params(named_params_dict):
   
    reference_params = next(iter(named_params_dict.values()))
    global_w_update = copy.deepcopy(reference_params) 

    for k in global_w_update.keys():
     
        parameter_values = [params[k] for params in named_params_dict.values()]
        
        # compute median
        aggregated_parameter = torch.median(torch.stack(parameter_values, dim=0), dim=0).values
        
        global_w_update[k] = aggregated_parameter
  
    return OrderedDict(global_w_update)


def federated_median_by_params(models_params):
    """Compute weighted average of model parameters and persistent buffers.
    Using state_dict of model, including persistent buffers like BN stats.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
    """
    # average_weights_dict = get_average_weight(weights)
    model_params = median_named_params(models_params)

    return model_params
