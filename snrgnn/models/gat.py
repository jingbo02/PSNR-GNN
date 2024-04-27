import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
import dgl
from .dataprocess import DataProcess
import pdb

class GAT(nn.Module):
    def __init__(self, nfeat:int, nhid:int, nclass:int, num_layers:int, num_heads, activation:str, norm:list, drop:list, residual:str):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.name = 'GAT_'+ residual
        self.hidden_list = []

        # Conv Layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(nfeat, nhid, num_heads = self.num_heads))
        for i in range(self.numLayers - 1):
            self.convs.append(GATConv(nhid, nhid, num_heads = self.num_heads))

        self.out_fc = nn.Linear(nhid, nclass)
        self.data_process = DataProcess(num_layers, nfeat, nhid, nclass, residual, drop, norm, activation)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, graph, x):
        h = self.data_process.drop(graph, x)
        h = self.convs[0](graph, h)
        h = torch.mean(h,dim=1)
        h = self.data_process.norm(h)
        h = self.data_process.activation(h)
        self.hidden_list.append(h)
        
        for i in range(1, self.num_layers):
            h = self.data_process.drop(graph, x)
            h = self.convs[i](graph, h)
            h = torch.mean(h,dim=1)
            h = self.data_process.residual(self.hidden_list, h, i)
            h = self.data_process.norm(h)
            h = self.data_process.activation(h)
            self.hidden_list.append(h)

        h = self.out_fc(h)
        h = F.log_softmax(h, dim=1)
        return h


class GATppi(nn.Module):
    def __init__(self, nfeat:int, nhid:int, nclass:int, num_layers:int, activation:str, norm:list, drop:list, residual:str, num_node:int, args):
        super().__init__()
        self.residual = residual
        self.num_layers = num_layers
        self.name = 'GCN_'+ residual
        self.hidden_list = []
        self.heads = args.heads
        
        assert self.num_layers == len(self.heads), "The number of attention heads is not equal to the number of layers."
        
        print("Right!!")
        # Conv Layers
        self.convs = nn.ModuleList()
        self.in_fc = nn.Linear(nfeat, nhid)

        self.convs.append(GATConv(nfeat, nhid, num_heads = self.heads[0]))
        for i in range(1, self.num_layers - 1):
            self.convs.append(GATConv(nhid * self.heads[i - 1], nhid, num_heads = self.heads[i]))  
        self.convs.append(GATConv(nhid * self.heads[-2], nclass, num_heads = self.heads[-1]))
        
        self.data_process = DataProcess(num_layers, nfeat, nhid, nclass, residual, drop, norm, activation, num_node, args)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.residual not in ['snr']:
            self.in_fc.reset_parameters()

    def forward(self, graph, h):
        self.hidden_list = []
        graph, h = self.data_process.drop(graph, h, self.training)
        # pdb.set_trace()
        h = self.convs[0](graph, h)
        if self.num_layers == 1:
            h = h.mean(1)
        else:
            h = h.flatten(1)        
        h = self.data_process.normalization(h)
        h = self.data_process.activation(h)
        self.hidden_list.append(h)

        for i in range(1, self.num_layers - 1):
            graph, h = self.data_process.drop(graph, h, self.training)
            h = self.convs[i](graph, h)
            h = h.flatten(1)
            h = self.data_process.residual(self.hidden_list, h, i, graph)
            h = self.data_process.normalization(h)
            h = self.data_process.activation(h)
            self.hidden_list.append(h)
        
        if self.num_layers > 1:
            graph, h = self.data_process.drop(graph, h, self.training)
            h = self.convs[-1](graph, h)
            h = h.mean(1)
            h = self.data_process.normalization(h)
        
        return h