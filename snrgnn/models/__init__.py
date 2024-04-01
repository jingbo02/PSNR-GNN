import torch.nn.functional as F
from dgl import DropEdge
import torch
import torch.nn as nn
import gcn, gat, sage

class DataProcess(nn.Module):
    def __init__(self, numlayers, nfeat, nhid, nclass, Residual, Drop, Norm, Activation):
        self.numlayers = numlayers
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.Residual = Residual
        self.Drop = Drop
        self.Norm = Norm
        self.Activation = Activation

        if self.residual == "dense":
            self.linear = nn.ModuleList()
            for i in range(self.num_layers):
                self.linear.append(nn.Linear((i+2)*nhid,nhid))
        if self.residual == "jk":
            self.linear = nn.Linear(self.num_layers * nhid,nhid)
        
        self.norm = nn.ModuleList()
        if 'batch' in Norm:
            self.norm.append(nn.BatchNorm1d(nhid))
        if 'layer' in Norm:
            self.norm.append(nn.LayerNorm(nhid))

    def residual(self, hidden_list, x, layer):
        if self.mode == 'res':
            return hidden_list[-1] + x
        if self.mode == 'initres':
            return hidden_list[0] + x
        if self.mode == 'dense':
            for i in range(self.numlayers):
                x = torch.cat([x, hidden_list[i]], dim=1)
            return self.liear[layer](x)
        if self.mode == 'jk':
            if layer == self.numlayers:
                for i in range(1, self.numlayers):
                    x = torch.cat([x, hidden_list[i]], dim=1)
                return self.linear(x)

    def drop(self, g, x):
        """
        drop_list: [drop_out_ratio,drop_edge_ratio]
        """
        if self.drop_list[0] != 0.0:
            x = F.dropout(x, p=self.drop[0], training=self.training)
        if self.drop_list[1] != 0.0:
            x = DropEdge(self.drop[1])(g)
        return x
    
    def activation(self, x):
        if self.Activation == 'relu':
            return F.relu(x)
        if self.Activation == 'elu':
            return F.elu(x)
        if self.Activation == 'leaky_relu':
            return F.leaky_relu(x)
        if self.Activation == 'sigmoid':
            return F.sigmoid(x)
        if self.Activation == 'tanh':       
            return F.tanh(x)
        if self.Activation == 'identity':
            return x
        if self.Activation == 'gelu':
            return F.gelu(x)
        

    def norm(self, x):
        for norm in self.norm:
            x = norm(x)
        return x
        


class BuildModel():
    def __init__(self, model_name, nfeat:int, nhid:int, nclass:int, num_layers:int, activation:str, norm:list, drop:list, residual:str):
        self.model_name = model_name
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.num_layers = num_layers
        self.norm = norm
        self.drop = drop
        self.activation = activation
        self.residual = residual

    def build(self):
        if self.model_name == 'gcn':
            return gcn.GCN(self.nfeat, self.nhid, self.nclass, self.num_layers, self.activation, self.norm, self.drop, self.residual)
        if self.model_name == 'gat':
            #TODO            
            pass
        if self.model_name == 'sage':
            #TODO
            pass