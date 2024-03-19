import torch.nn.functional as F
from dgl import DropEdge

class DataProcess():
    def __init__(self,numlayers,nfeat,nhid,nclass):
        self.numlayers = numlayers
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass

    def residual(self, mode, hidden_list):
        
        pass

    def drop(self,drop_list,g,x):
        """
        drop_list: [drop_out_ratio,drop_edge_ratio]
        """
        if drop_list[0] != 0.0:
            x = F.dropout(x, p=drop_list[0], training=self.training)
        if drop_list[1] != 0.0:
            x = DropEdge(drop_list[1])(g)
        return x
    
    def activation(self,type,x):
        pass

    def norm(self,type,x):
        pass