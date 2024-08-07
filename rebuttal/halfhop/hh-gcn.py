import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from halfhop import HalfHop
from createPyGdataset import MyOwnDataset
import os.path as osp
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GATConv  # noqa
import numpy as np
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    default=True)
parser.add_argument('--patience', type=int, default=1000)
parser.add_argument('--nhid',type=int,default=128)
parser.add_argument('--times',type=int,default=5)
parser.add_argument('--d',type=str,default='cora')
parser.add_argument('--layer',type=int,default=2)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--device',type=str,default='0')
parser.add_argument('--bb',type=str,default='gcn')
args = parser.parse_args()

# 训练使用的数据集
data0 = MyOwnDataset(root='../DropMessage/PyGdata/'+args.d)
# apply halfhop
transform = HalfHop(alpha=0.1,p=0.1)
save_path = '../default_splits/' + args.d + '_splits.npy'
data = np.load(save_path)
labels_matrix = data
labels = labels_matrix[0]
val_mask = labels == 1
test_mask = labels == 2
data0.x[val_mask] = 0.0
data0.x[test_mask] = 0.0
data0 = transform(data0)


class Net(torch.nn.Module):
    def __init__(self,type='gcn'):
        super(Net, self).__init__()
        self.conv = nn.ModuleList()

        if type == 'gcn':
            self.conv1 = GCNConv(data0.num_features, args.nhid, cached=True,
                                normalize=args.use_gdc)
            self.conv.append(self.conv1)
            for _ in range(args.layer - 2):
                self.conv.append(GCNConv(args.nhid, args.nhid, cached=True,
                                        normalize= args.use_gdc))
            self.conv2 = GCNConv(args.nhid, data0.num_classes, cached=True,
                                normalize=args.use_gdc)
            self.conv.append(self.conv2)
        elif type == 'gat':
            self.conv1 = GATConv(data0.num_features, args.nhid)
            self.conv.append(self.conv1)
            for _ in range(args.layer - 2):
                self.conv.append(GATConv(args.nhid, args.nhid))
            self.conv2 = GATConv(args.nhid, data0.num_classes)
            self.conv.append(self.conv2)
        
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()
        
    def forward(self):
        global data0
        x, edge_index = data0.x, data0.edge_index
        for conv in self.conv:
            x = F.dropout(x, p=0.5,training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda:'+ args.device if torch.cuda.is_available() else 'cpu')
data0 =data0.to(device)


@torch.no_grad()
def test(time):
    model.eval()
    save_path = '../default_splits/' + args.d + '_splits.npy'
    # with np.load(save_path) as data:
    data = np.load(save_path)
    labels_matrix = data
    labels = labels_matrix[time]
    if args.d == 'ogbn-papers100M':
        train_mask = labels == 3
    else:
        train_mask = labels == 0
        val_mask = labels == 1
        test_mask = labels == 2
    logits, accs = model(), []
    logits = logits[~data0.slow_node_mask]
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data0.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


trainResult = []
valResult = []
testResult = []
for time in range(args.times):
    model=Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = best_test = best_train = best_epoch=0
    badCounter = 0
    for epoch in range(1, 1001):
        model.train()
        optimizer.zero_grad()
        save_path = '../default_splits/' + args.d + '_splits.npy'
        # with np.load(save_path) as data:
        data = np.load(save_path)
        labels_matrix = data
        labels = labels_matrix[time]
        if args.d == 'ogbn-papers100M':
            train_mask = labels == 3
        else:
            train_mask = labels == 0
        val_mask = labels == 1
        test_mask = labels == 2
        y=model()
        # get rid of slow nodes
        y = y[~data0.slow_node_mask]
        F.nll_loss(y[train_mask], data0.y[train_mask].long()).backward()
        optimizer.step()
        train_acc, val_acc, test_acc = test(time)
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, val_acc, test_acc))
        if val_acc > best_val:
            best_train = train_acc
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch
        else:
            badCounter += 1
        if badCounter == args.patience:
            break

    print("Optimization Finished!")
    print(f"Best epoch: {best_epoch:03d}, best train: {best_train:.4f}, best val: {best_val:.4f}, best_test: {best_test:.4f}")
    trainResult.append(best_train)
    valResult.append(best_val)
    testResult.append(best_test)

avg_train = np.mean(trainResult)
train_std = np.std(trainResult)
avg_val = np.mean(valResult)
val_std = np.std(valResult)
avg_test = np.mean(testResult)
test_std = np.std(testResult)
print("ALL Finished!")
print(f"avg_train: {avg_train:.4f}, avg_val: {avg_val:.4f}, avg_test: {avg_test:.4f}, test_std: {test_std:.4f}")

import csv
with open('mv.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["DataSet", "Layer","TrainAvg", "TrainStd", "ValAvg", "ValStd", "TestAvg", "TestStd",'model type'])
    writer.writerow(
        [args.d, args.layer, avg_train, train_std, avg_val, val_std, avg_test, test_std,args.bb])

