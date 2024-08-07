


import torch.nn.functional as F
from dgl import DropEdge
import torch
import torch.nn as nn
from .gcn import GCN

        


class BuildModel():
    def __init__(self, model_name, nfeat:int, nhid:int, nclass:int, num_layers:int, activation:str, norm:list, drop:list, residual:str, num_node:int):
        self.model_name = model_name
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.num_layers = num_layers
        self.norm = norm
        self.drop = drop
        self.activation = activation
        self.residual = residual
        self.num_node = num_node

    def build(self, args):
        if self.model_name == 'gcn':
            return gcn.GCN(self.nfeat, self.nhid, self.nclass, self.num_layers, self.activation, self.norm, self.drop, self.residual, self.num_node, args)

        