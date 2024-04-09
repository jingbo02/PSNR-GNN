import torch.nn.functional as F
from dgl import DropEdge
import torch
import torch.nn as nn
import dgl

class SNRModule(nn.Module):
    def __init__(self, nodes_num, args):
        super().__init__()
        # Set the mean and standard deviation to learnable parameters
        self.nodes_num =  nodes_num
        if not args.randn_init:
            mean = torch.zeros(self.nodes_num,1)
            std = torch.ones(self.nodes_num,1)
        else:
            mean = F.sigmoid(torch.randn(self.nodes_num,1))
            std = F.sigmoid(torch.randn(self.nodes_num,1))

        self.mean = nn.Parameter(torch.FloatTensor(mean))
        self.std = nn.Parameter(torch.FloatTensor(std))

    def forward(self, input):
        x  = torch.randn(self.nodes_num,1)
        y = torch.ones(self.nodes_num,1)
        x = x.to(self.mean.device)
        y = y.to(self.mean.device)
        return input * F.sigmoid(x*self.std + y*self.mean)
        #TODO : check if calculation is correct



class DataProcess(nn.Module):
    def __init__(self, numlayers, nfeat, nhid, nclass, Residual, Drop, Norm, Activation, nodes_num, args):
        super().__init__()
        self.numlayers = numlayers
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.res_type = Residual
        self.drop_list = Drop
        self.act_type = Activation

        if self.res_type == "dense":
            self.linear = nn.ModuleList()
            for i in range(self.num_layers):
                self.linear.append(nn.Linear((i+2)*nhid,nhid))
        if self.res_type == "jk":
            self.linear = nn.Linear(self.num_layers * nhid,nhid)
        if self.res_type == "snr":
            self.SnrList = nn.ModuleList()
            for i in range(self.num_layers):
                self.SnrList.append(SNRModule(nodes_num, args))


        self.NormList = nn.ModuleList()
        if 'batch' in Norm:
            self.NormList.append(nn.BatchNorm1d(nhid))
        if 'layer' in Norm:
            self.NormList.append(nn.LayerNorm(nhid))

    def residual(self, hidden_list, x, layer):
        if self.res_type == 'res':
            return hidden_list[-1] + x
        if self.res_type == 'initres':
            return hidden_list[0] + x
        if self.res_type == 'dense':
            for i in range(self.numlayers):
                x = torch.cat([x, hidden_list[i]], dim=1)
            return self.liear[layer](x)
        if self.res_type == 'jk':
            if layer == self.numlayers:
                for i in range(1, self.numlayers):
                    x = torch.cat([x, hidden_list[i]], dim=1)
                return self.linear(x)
        
        if self.res_type == 'snr':
            return hidden_list[0] + self.SnrList[layer](x - hidden_list[0])

    def drop(self, g, x, training):
        """
        drop_list: [drop_out_ratio,drop_edge_ratio]
        """
        if self.drop_list[0] != 0.0:
            x = F.dropout(x, p=self.drop[0], training=training)
        if self.drop_list[1] != 0.0 and training:
            g = DropEdge(self.drop[1])(g)
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
        

    def norm(self, x):
        for norm in self.NormList:
            x = norm(x)
        return x