from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset,
    CoauthorCSDataset,
    AmazonCoBuyPhotoDataset,
    CoauthorPhysicsDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os

GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset,
    "coauther_cs": CoauthorCSDataset,
    'amazon_photo': AmazonCoBuyPhotoDataset,
    'coauther_phy':CoauthorPhysicsDataset
}


def load_inductive_dataset():
    batch_size = 1
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')
    train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    g = train_dataset[0]
    num_classes = train_dataset.num_labels
    num_features = g.ndata['feat'].shape[1]
    return train_dataloader, valid_dataloader, test_dataloader, num_classes, num_features


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def split_datasets(label):
    split_list = []
    num_split = 10
    data_list = [i for i in range(len(label))]

    label_ = np.array(label)
    num_class = len(Counter(label_))
    print("The number of class: ", num_class)
    for random_state in range(num_split): 
        
        x_train, x_temp, y_train, y_temp = train_test_split(data_list, label, train_size=20*num_class, stratify=label, random_state=random_state)
        x_val, x_tmp, y_val, y_tmp = train_test_split(x_temp, y_temp, train_size=500, stratify=y_temp, random_state=random_state)
        x_test, x_t, y_test, y_t = train_test_split(x_tmp, y_tmp, train_size=1000, stratify=y_tmp, random_state=random_state)
        print(f"No.{random_state} split:")
        print("The length of x_train: ",len(x_train))
        print("The length of x_val: ",len(x_val))
        print("The length of x_test: ",len(x_test))
        print("The summation: ",len(x_train)+len(x_val)+len(x_test))

        split_libel = torch.tensor(np.full(len(data_list), 3, dtype=int))
        split_libel[x_train] = 0  
        split_libel[x_test] = 1  
        split_libel[x_val] = 2   
        
        split_list.append(split_libel)
    splits = torch.stack(split_list, dim = 0)
    return splits.numpy()

def load_dataset(dataset_name, args):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()
        
        
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.fadjull((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    num_nodes = graph.num_nodes()
    return graph, (num_features, num_classes, num_nodes)

