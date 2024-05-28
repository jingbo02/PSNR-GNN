import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
import dgl
from .dataprocess import DataProcess
import pdb

class GAT(nn.Module):
    def __init__(self, nfeat:int, nhid:int, nclass:int, num_layers:int, activation:str, norm:list, drop:list, residual:str, num_node, args):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = args.num_heads
        self.name = 'GAT_'+ residual
        self.residual = residual
        self.hidden_list = []

        # Conv Layers
        self.convs = nn.ModuleList()
        self.in_fc = nn.Linear(nfeat, nhid )
        if self.residual not in ['psnr']:
            self.convs.append(GATConv(nhid , nhid, num_heads = self.num_heads))
        else:
            self.convs.append(GATConv(nfeat, nhid, self.num_heads))
        for i in range(self.num_layers - 1):
            self.convs.append(GATConv(nhid , nhid, self.num_heads))

        self.out_fc = nn.Linear(nhid , nclass)
        self.data_process = DataProcess(num_layers, nfeat, nhid , nclass, residual, drop, norm, activation, num_node, args)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.out_fc.reset_parameters()
        if self.residual not in ['psnr']:
            self.in_fc.reset_parameters()

    def forward(self, graph, h):
        self.hidden_list = []
        if self.residual not in ['psnr']:
            graph, h = self.data_process.drop(graph, h, self.training)
            h = self.in_fc(h)
            h = self.data_process.normalization(h)
            h = self.data_process.activation(h)
            self.hidden_list.append(h)

        graph, h = self.data_process.drop(graph, h, self.training)
        h = self.convs[0](graph, h)
        h = h.mean(1)
        if self.residual not in ['psnr']:
            h = self.data_process.residual(self.hidden_list, h, 0,graph)
        h = self.data_process.normalization(h)
        h = self.data_process.activation(h)
        self.hidden_list.append(h)

        for i in range(1, self.num_layers):
            graph, h = self.data_process.drop(graph, h, self.training)
            h = self.convs[i](graph, h)
            h = h.mean(1)
            h = self.data_process.residual(self.hidden_list, h, i, graph)
            h = self.data_process.normalization(h)
            self.hidden_list.append(h)

        h = self.out_fc(h)
        h = F.log_softmax(h, dim=1)
        return h