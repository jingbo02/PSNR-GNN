import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import warnings
import dgl
from ogb.nodeproppred import DglNodePropPredDataset

from dgl.data import WikiCSDataset
from dgl.data import FlickrDataset
from dgl.data import YelpDataset
from dgl.data import RedditDataset
from dgl.data import CoraFullDataset
from dgl.data import CoraGraphDataset
from dgl.data import CiteseerGraphDataset
from dgl.data import PubmedGraphDataset
import numpy as np



class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        if 'ogb' in self.root:
            dataset1 = DglNodePropPredDataset(self.root.split('/')[-1])
        elif 'wikics' in self.root:
            dataset1 = WikiCSDataset()
        elif 'corafull' in self.root:
            dataset1 = CoraFullDataset()
        elif 'flickr' in self.root:
            dataset1 = FlickrDataset()
        elif 'yelp' in self.root:
            dataset1 = YelpDataset()
        elif 'reddit' in self.root:
            dataset1 = RedditDataset()
        elif 'cora' in self.root:
            dataset1 = CoraGraphDataset()
        elif 'citeseer' in self.root:
            dataset1 = CiteseerGraphDataset()
        elif 'pubmed' in self.root:
            dataset1 = PubmedGraphDataset()

        if 'ogbn' in self.root:
            g, label = dataset1[0]
            label = label.reshape(-1)
        else:
            g = dataset1[0]
            label = g.ndata['label'].tolist()

        x = torch.tensor(g.ndata['feat'], dtype=torch.float)
        y = torch.tensor(label, dtype=torch.float)
        feat = g.ndata['feat']
        num_features= int(len(feat[0]))
        if 'yelp' in self.root:
            print("yelp_time")
            print(label[0])
            print(len(label[0]))
            num_classes = int(len(label[0]))
        else:
            print("other_time")
            print(label[0])
            num_classes = int(max(label)) + 1
        num_nodes = int(len(feat))

        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        edge_index  = torch.stack((g.edges()[0], g.edges()[1]), dim=0)

        data = Data(x=x, edge_index=edge_index, y=y, num_features=num_features, num_classes=num_classes,num_nodes=num_nodes)
        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        self.device = device
        return self
