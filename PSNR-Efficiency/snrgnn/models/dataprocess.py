

import sys, os
import os
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
from snrgnn.utils import p





import torch.nn.functional as F
from dgl import DropEdge
import torch
import torch.nn as nn
import dgl
import pdb
from dgl.nn import GATConv,GraphConv,SAGEConv   
import math
import torch

import torch.nn as nn

class GATencoder(nn.Module):
    def __init__(self, n_hid, out_hid, num_head):
        super().__init__()
        self.encoder = GATConv(n_hid, out_hid, num_heads = num_head)
        self.linear = nn.Linear(out_hid * num_head, 2)
    def forward(self, graph, x):
        x = self.encoder(graph, x)
        x = x.flatten(1)
        # pdb.set_trace()
        return self.linear(x)
        
    

class SNRModule(nn.Module):
    def __init__(self, nodes_num, n_hid, args):
        super().__init__()
        self.nodes_num =  nodes_num
        self.coef_encoder_type = args.coef_encoder
        self.if_var = args.if_var

        if args.coef_encoder == 'gat':
            self.coef_encoder = GATConv(n_hid, 2, num_heads = 1)
            # self.coef_encoder = GATencoder(n_hid, 2, 4)
                
        if args.coef_encoder == 'sage':
            self.coef_encoder = SAGEConv(n_hid, 2, 'gcn')
        if args.coef_encoder == 'gcn':
            self.coef_encoder = GraphConv(n_hid, 2)
        if args.coef_encoder == 'mlp':
            self.coef_encoder = nn.Sequential(
                nn.Linear(n_hid, n_hid),
                nn.ReLU(),
                nn.Linear(n_hid, 2)
            ) 

        self.d_model = n_hid
        self.max_seq_len = args.n_layers +  3

        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float()
                             * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe
        self.pe_coff = nn.Parameter(torch.tensor(0.1))
        self.layer_emb = args.layer_emb

        
        
        
        
    def forward(self, graph, input, t):

        # for inductive 
        self.nodes_num = input.shape[0]

        x  = torch.randn(self.nodes_num, 1, device = graph.device)
        y = torch.ones(self.nodes_num, 1, device = graph.device)
        

        if self.layer_emb:
            input = input + self.pe_coff*self.pe[t+1].to(input.device)

        if self.coef_encoder_type == 'mlp':
            coef = self.coef_encoder(input)
        else:
            coef = self.coef_encoder(graph, input)
            if self.coef_encoder_type == 'gat':
                coef = torch.mean(coef, dim=1)

        std = F.relu(coef[:,0])
        mean = F.relu(coef[:,1])

        std = std.view(-1,1)
        mean = mean.view(-1,1)
        if self.training or self.if_var:
            return input * (F.sigmoid(x * std + y*mean))
        else:
            return input * (F.sigmoid(y*mean))
        
        
        


class DataProcess(nn.Module):
    def __init__(self, nlayers, nfeat, nhid, nclass, Residual, Drop, Norm, Activation, nodes_num, args):
        super().__init__()
        self.nlayers = nlayers
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.res_type = Residual
        self.drop_list = Drop
        self.act_type = Activation

        if self.res_type == "dense":
            self.linear = nn.ModuleList()
            for i in range(self.nlayers):
                self.linear.append(nn.Linear((i+2)*nhid,nhid))
        if self.res_type == "jk":
            self.linear = nn.Linear((self.nlayers+1) * nhid,nhid)
        if self.res_type in ["snr"]:
            self.SnrList = nn.ModuleList()
            self.SnrList.append(SNRModule(nodes_num, nhid, args))



        self.NormList = nn.ModuleList()
        if 'batch' in Norm:
            self.NormList.append(nn.BatchNorm1d(nhid))
        if 'layer' in Norm:
            self.NormList.append(nn.LayerNorm(nhid))

    def residual(self, hidden_list, x, layer, graph):
        # import pdb; pdb.set_trace()
        if self.res_type == 'res':
            # print('res')
            return hidden_list[-1] + x
        if self.res_type == 'init_res':
            return hidden_list[0] + x
        if self.res_type == 'dense':
            for i in range(layer+1):
                x = torch.cat([x, hidden_list[i]], dim=1)
            return self.linear[layer](x)
        if self.res_type == 'jk':
            if layer == (self.nlayers - 1):
                for i in range(self.nlayers):
                    x = torch.cat([x, hidden_list[i]], dim=1)
                return self.linear(x)
            else:
                return x
        
        if self.res_type in ['snr']:
            return hidden_list[0] + self.SnrList[0](graph, hidden_list[0] - x, layer - 1)
        
        if self.res_type == 'none':
            return x

        
    def drop(self, g, x, training):
        """
        drop_list: [drop_out_ratio,drop_edge_ratio]
        """
        if self.drop_list[0] != 0.0:
            x = F.dropout(x, p=self.drop_list[0], training=training)
        if self.drop_list[1] != 0.0 and training:
            g = DropEdge(self.drop_list[1])(g)
            g = dgl.add_self_loop(g)                        #TODO

        return g, x
    
    def activation(self, x):
        if self.act_type == 'relu':
            return F.relu(x)
        if self.act_type == 'elu':
            return F.elu(x)
        if self.act_type == 'leaky_relu':
            return F.leaky_relu(x)
        if self.act_type == 'sigmoid':
            return F.sigmoid(x)
        if self.act_type == 'tanh':       
            return F.tanh(x)
        if self.act_type == 'identity':
            return x
        if self.act_type == 'gelu':
            return F.gelu(x)
        if self.act_type == 'silu':
            return F.silu(x)
        

    def normalization(self, x):
        for norm in self.NormList:
            x = norm(x)
        return x