import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pdb


from psnrgnn.utils import (
    create_optimizer,
    set_random_seed,
    accuracy,
    evaluate,
    p,
)

from psnrgnn.datasets.dataset import load_dataset, split_datasets
from psnrgnn.models import BuildModel

import sys, os
import os
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))



def train(model, graph, optimizer, max_epoch, if_mv, if_early_stop):

    x = graph.ndata["feat"]

    label = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    
    if if_mv :
        x[val_mask] = 0
        x[test_mask] = 0
        
    epoch_iter = tqdm(range(max_epoch))

    best_acc = 0
    final_acc = 0
    cnt = 0
    for epoch in epoch_iter:
        model.train()
        output = model(graph, x)
        loss_train = F.nll_loss(output[train_mask], label[train_mask])

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            acc_train = accuracy(output[train_mask], label[train_mask])
            acc_val = evaluate(model, graph, x, val_mask)
            acc_test = evaluate(model, graph, x, test_mask)

        if acc_val > best_acc :
            best_acc = acc_val
            final_acc = acc_test
        else:
            cnt += 1
        if cnt > 100 and if_early_stop:
            break

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss_train.item():.4f} train_acc: {acc_train:.4f} val_acc: {acc_val:.4f} test_acc: {acc_test:.4f}  final_acc: {final_acc:.4f} ")
    return best_acc, final_acc 


def main(args):
    device = "cuda:" + str(args.device) if args.device >= 0 else "cpu"
    graph, (num_features, num_classes, num_node) = load_dataset(args.dataset, args)
    args.num_features = num_features 

    # return
    graph = graph.to(device)    
    acc_list = []
    val_acc_list = []
    
    for i, seed in enumerate(args.seeds):
        set_random_seed(seed)        
        model = BuildModel(
            args.backbone, 
            num_features, 
            args.n_hid, 
            num_classes, 
            args.n_layers, 
            args.activation, 
            args.norm, 
            args.drop, 
            args.residual_type, 
            num_node
        ).build(args)
        
        model = model.to(device)
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

        best_acc, final_acc = train(model, graph, optimizer, args.max_epoch, args.if_mv, args.if_early_stop)
        val_acc_list.append(best_acc.cpu())
        acc_list.append(final_acc.cpu())

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    val_acc, val_acc_std = np.mean(val_acc_list), np.std(val_acc_list)

    return final_acc, final_acc_std, val_acc, val_acc_std




