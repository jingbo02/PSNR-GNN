import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pdb


from snrgnn.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
    accuracy,
    evaluate
)
from snrgnn.datasets.dataset import load_dataset, split_datasets
# from graphmae.evaluation import node_classification_evaluation
from snrgnn.models import BuildModel

import wandb


import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)



def train(model, graph, optimizer, max_epoch, scheduler, num_classes, logger = None, save_model = False):
    # logging.info("start training..")

    x = graph.ndata["feat"]
    label = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    epoch_iter = tqdm(range(max_epoch))

    best_acc = 0

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

        if acc_val > best_acc and save_model:
            best_acc = acc_val


        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss_train.item():.4f} train_acc: {acc_train:.4f} val_acc: {acc_val:.4f} test_acc: {acc_test:.4f}")






def main(args):
    device = "cuda:" + str(args.device) if args.device >= 0 else "cpu"
    # TO-DO
    seeds = args.seeds
    dataset_name = args.dataset

    optim_type = args.optimizer 

    lr = args.lr
    weight_decay = args.weight_decay
    logs = args.logging

    graph, (num_features, num_classes, num_node) = load_dataset(dataset_name, args)

    args.num_features = num_features

    acc_list = []

    if args.split_dataset:
        if args.loda_split:
            filename = os.path.join(args.pre_split_path, args.datase + '_splits.npz')
            splits_list = np.load(filename)
        else:
            splits_list = split_datasets(graph.ndata["label"], args.num_split)
            if args.save_split:
                if not os.path.exists(args.pre_split_path):
                    os.makedirs(args.pre_split_path)
                np.savez(os.path.join(args.pre_split_path, args.dataset + '_splits.npz'), splits_list)
    else:
        splits_list = np.zeros(num_node, dtype=int)
        splits_list[graph.ndata["train_mask"]] = 0  
        splits_list[graph.ndata["test_mask"]] = 1  
        splits_list[graph.ndata["val_mask"]] = 2   

    graph = graph.to(device)    

    for i in range(splits_list.shape[0]):
        split = splits_list[i]
        train_idx = torch.tensor(np.where(split == 0, True, False)).to(device)
        test_idx = torch.tensor(np.where(split == 1, True, False)).to(device)
        val_idx = torch.tensor(np.where(split == 2, True, False)).to(device)
        
        # print(type(graph.ndata["train_mask"]), type(train_idx))

        graph.ndata["train_mask"] = train_idx
        graph.ndata["test_mask"] = test_idx
        graph.ndata["val_mask"] = val_idx

        for j, seed in enumerate(seeds):
            # print(f"####### Run {i} for seed {seed}")
            set_random_seed(seed)

            model = BuildModel(args.backbone, num_features, args.n_hid, num_classes, args.n_layers, args.activation, args.norm, args.drop, args.residual_type, num_node).build(args)
            model = model.to(device)

            optimizer = create_optimizer(optim_type, model, lr, weight_decay)
            print(next(model.parameters()).device)


            train(model, graph, optimizer, args.max_epoch,  num_classes, logger, args.save_model)

            
            x = graph.ndata["feat"]
            test_mask = graph.ndata['test_mask']
            final_acc = evaluate(model, graph, x, test_mask)
            acc_list.append(final_acc.cpu())

        final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
        print(f"# spilt: {i}, final_acc: {final_acc:.4f}±{final_acc_std:.4f}")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)





