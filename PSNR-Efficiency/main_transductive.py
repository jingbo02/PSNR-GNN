import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pdb


from snrgnn.utils import (
    create_optimizer,
    set_random_seed,
    accuracy,
    evaluate,
    p,
)

from snrgnn.datasets.dataset import load_dataset, split_datasets
from snrgnn.models import BuildModel



import sys, os
import os
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
from snrgnn.utils import p


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

        start_time = time.time()
        output = model(graph, x)


        end_time = time.time()
        # print(end_time - start_time)

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
            # cnt += 1

        else:
            cnt += 1
        # if cnt > 0 and if_early_stop:
        if cnt > 500 and if_early_stop:
            break
        # print(torch.cuda.memory_allocated()/(1024**2))
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss_train.item():.4f} train_acc: {acc_train:.4f} val_acc: {acc_val:.4f} test_acc: {acc_test:.4f}  final_acc: {final_acc:.4f} ")
    return best_acc, final_acc 


def main(args):

    device = "cuda:" + str(args.device) if args.device >= 0 else "cpu"
    graph, (num_features, num_classes, num_node) = load_dataset(args.dataset, args)
    args.num_features = num_features

    if args.loda_split:
        filename = os.path.join(args.pre_split_path, args.dataset + '_splits.npy')
        splits_list = np.load(filename)
    else:
        splits_list = split_datasets(graph.ndata["label"])
        if not os.path.exists(args.pre_split_path):
            os.makedirs(args.pre_split_path)
        np.save(os.path.join(args.pre_split_path, args.dataset + '_splits.npy'), splits_list)        

    graph = graph.to(device)    
    set_random_seed(args.seed)
    acc_list = []
    val_acc_list = []
    
    # for i in range(splits_list.shape[0]):
    i = 0
    split = splits_list[i]
    train_idx = torch.tensor(np.where(split == 0, True, False)).to(device)
    test_idx = torch.tensor(np.where(split == 1, True, False)).to(device)
    val_idx = torch.tensor(np.where(split == 2, True, False)).to(device)

    graph.ndata["train_mask"] = train_idx
    graph.ndata["test_mask"] = test_idx
    graph.ndata["val_mask"] = val_idx

    model = BuildModel(args.backbone, num_features, args.n_hid, num_classes, args.n_layers, args.activation, args.norm, args.drop, args.residual_type, num_node).build(args)
    model = model.to(device)
    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    print("Parameter Num:",sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    p.print('Residual:'+ args.residual_type+ "- Before Train:", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
    p.print('Residual:'+ args.residual_type+ "- Before Train:", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
    best_acc, final_acc = train(model, graph, optimizer, args.max_epoch, args.if_mv, args.if_early_stop)
    p.print(args.backbone + 'Residual:'+ args.residual_type+ "- Training Time:", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)

    val_acc_list.append(best_acc.cpu())
    acc_list.append(final_acc.cpu())

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    val_acc, val_acc_std = np.mean(val_acc_list), np.std(val_acc_list)

    return final_acc, final_acc_std, val_acc, val_acc_std




