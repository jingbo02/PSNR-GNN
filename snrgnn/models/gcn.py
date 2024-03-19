import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DropEdge
from dgl.nn import GraphConv
import dgl
import DataProcess

class GCN(nn.Module):
    def __init__(self, nfeat:int, nhid:int, nclass:int, num_layers:int,activation:str,norm:list,drop:list,residual:str):
        super().__init__()

        self.num_layers = num_layers
        self.activation = activation
        self.norm = norm
        self.drop = drop
        self.residual = residual
        self.name = 'GCN_'+ residual

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(nfeat, nhid))
        for i in range(self.numLayers - 1):
            self.convs.append(GraphConv(nhid, nhid))
        self.out_fc = nn.Linear(nhid, nclass)

        self.data_process = DataProcess(num_layers,nfeat,nhid,nclass)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, graph, x):
        h = self.data_process.drop()



        h = F.dropout(inputs, p=self.dropout, training=self.training)
        h = self.convs[0](graph, h)
        if self.batchNorm:
            h = self.bns[0](h)
        if self.layerNorm:
            h = self.ln(h)
        h = self.activationFunc(h)
        for i in range(1, self.numLayers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.convs[i](graph, h)
            if self.batchNorm:
                h = self.bns[i](h)
            if self.layerNorm:
                h = self.ln(h)
            h = self.activationFunc(h)
        h = self.out_fc(h)
        return h

    