from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
from sklearn.metrics import f1_score


from snrgnn.datasets.dataset import load_inductive_dataset
from snrgnn.models import BuildModel
from snrgnn.utils import (
    create_optimizer,
    set_random_seed,
)
import wandb
import numpy as np
from tqdm import tqdm
import pdb
import dgl

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def output_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Gradients for {name}: {param.grad}")


def train(model, loaders, optimizer, max_epoch, device):
    train_loader, val_loader, test_loader = loaders

    epoch_iter = tqdm(range(max_epoch))

    best_acc = 0
    final_acc = 0
    cnt = 0
    loss_func = torch.nn.BCEWithLogitsLoss()
    

    for epoch in epoch_iter:
        # Data preparation
        model.train()
        loss_list = []

        for batch_id, batched_graph in enumerate(train_loader):
        # for subgraph in train_loader:
            # pdb.set_trace()
            batched_graph = batched_graph.to(device)

            x = batched_graph.ndata["feat"].float()
            label = batched_graph.ndata['label'].float()
            # pdb.set_trace()
            output = model(batched_graph, x)


            loss_train = loss_func(output, label)
            loss_list.append(loss_train.item())
            
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        with torch.no_grad():
            loss_train_mean = np.mean(loss_list)
            acc_train = evaluate_ppi(model, train_loader, device)
            acc_val = evaluate_ppi(model, val_loader, device)
            acc_test = evaluate_ppi(model, test_loader, device)

        if acc_val > best_acc :
            best_acc = acc_val
            final_acc = acc_test
        else:
            cnt += 1

        # if cnt > 100:
        #     break

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss_train_mean:.4f} train_acc: {acc_train:.4f} val_acc: {acc_val:.4f} test_acc: {acc_test:.4f}  final_acc: {final_acc:.4f} ")
    return best_acc, final_acc 




def accuracy_ppi(output, labels):
    pred = np.where(output.cpu().numpy()>0.5, 1, 0)
    acc = f1_score(labels.cpu().numpy(), pred, average='micro')
    return acc

def evaluate_ppi(model, loader, device):
    model.eval()
    acc_list = []
    for subgraph in loader:
        subgraph = subgraph.to(device)
        x = subgraph.ndata["feat"]
        label = subgraph.ndata['label']
        output = model(subgraph, x)
        acc = accuracy_ppi(output, label)
        acc_list.append(acc)
    return np.mean(acc_list)



def main():
    with wandb.init():
        args = wandb.config
        device = "cuda:" + str(args.device) if args.device >= 0 else "cpu"
        
        train_loader, val_loader, test_loader, num_classes, num_features = load_inductive_dataset()
        loaders = (train_loader, val_loader, test_loader)

        num_node = None
        model = BuildModel(args.backbone, num_features, args.n_hid, num_classes, args.n_layers, args.activation, args.norm, args.drop, args.residual_type, num_node).build(args)
        
        model.to(device)
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
        best_acc, final_acc = train(model, loaders, optimizer, args.max_epoch, device)
        
        wandb.log({"final_acc": final_acc, "val_acc": best_acc})
        wandb.finish()


