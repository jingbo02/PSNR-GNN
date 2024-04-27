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




class SNRModule(nn.Module):
    def __init__(self, nodes_num, args):
        super().__init__()
        self.nodes_num =  nodes_num
        self.coef_encoder_type = args.coef_encoder

        if args.coef_encoder == 'gat':
            self.coef_encoder = GATConv(args.n_hid, 2, num_heads = 1)
        if args.coef_encoder == 'sage':
            self.coef_encoder = SAGEConv(args.n_hid, 2, 'gcn')
        if args.coef_encoder == 'gcn':
            self.coef_encoder = GraphConv(args.n_hid, 2)
        if args.coef_encoder == 'mlp':
            self.coef_encoder = nn.Sequential(
                nn.Linear(args.n_hid, args.n_hid),
                nn.ReLU(),
                nn.Linear(args.n_hid, 2)
            ) 

        self.d_model = args.n_hid 
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

        x  = torch.randn(self.nodes_num,1)
        y = torch.ones(self.nodes_num,1)
        x = x.to(graph.device)
        y = y.to(graph.device)
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
        return input * (F.sigmoid(x * std + y*mean))


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
        if self.res_type == "snr":
            self.SnrList = nn.ModuleList()
            self.SnrList.append(SNRModule(nodes_num, args))


        self.NormList = nn.ModuleList()
        if 'batch' in Norm:
            self.NormList.append(nn.BatchNorm1d(nhid))
        if 'layer' in Norm:
            self.NormList.append(nn.LayerNorm(nhid))

    def residual(self, hidden_list, x, layer, graph):
        if self.res_type == 'res':
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
        
        if self.res_type == 'snr':
            return hidden_list[0] + self.SnrList[0](graph, hidden_list[0] - x, layer-1)
        
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
            g = dgl.add_self_loop(g)

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