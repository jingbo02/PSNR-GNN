import os
import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np

import dgl

import torch
import torch.nn as nn
from torch import optim as optim
import wandb


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

def build_wandb_args():
    args = {}
    args["method"] = "grid"
    metric = {
        'name': 'avg_test',
        'goal': 'maximize'   
        }
    args['metric'] = metric
    args['parameters'] = {}
    args['parameters'].update({
        # Hyperparameters
        'backbone': {'value': 'gcn'},
        'n_layers': {'value': 2},
        'residual_type': {'value': 'snr'},
        'max_epoch': {'value': 500},
        'randn_init': {'value': False}, # Initialization for parameter of SNRModule
        'activation': {'value': 'elu'},
        'seeds': {'value': [2024]}, #TODO 固定seed
        'dataset': {'value': 'cora'},
        'split_dataset': {'value': False},
        'pre_split_path': {'value': './datasets/split_data'},                  
        'loda_split': {'value': False}, #TODO delete save_split
        'num_split': {'value': 5},
        'device': {'value': 1},
        'num_heads': {'value': 4}, # number of hidden attention heads
        'optimizer': {'value': 'adam'},
        'use_cfg': {'value': False}, # if load best config
        'logging': {'value': False},
        'log_path': {'value': './logging_data'},

        # Hyperparameters Under Optimization
        "n_hid": {'values': [64, 128, 256]},
        'lr': {'values': [1e-3, 1e-4]}, # learning rate
        'weight_decay': {'values': [5e-4]},
        'drop': {'value': [0.1,0.1]},
        'norm': {'value': []},
        })

    return args


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer



def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def evaluate(model, graph, feature, idx):
    model.eval()
    output =  model(graph, feature)
    labels = graph.ndata['label']
    acc_val = accuracy(output[idx], labels[idx])
    return acc_val