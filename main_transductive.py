import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F



from snrgnn.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from snrgnn.datasets.dataset import load_dataset
# from graphmae.evaluation import node_classification_evaluation
from snrgnn.models import BuildModel

import wandb


import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    # logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)
    label = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()
        output = model(graph, x)
        loss_train = F.nll_loss(output[train_mask], label[train_mask])
        acc_train = accuracy(output[train_mask], label[train_mask])
        acc_val = evaluate(model, graph, x, val_mask)
        acc_test = evaluate(model, graph, x, test_mask)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss_train.item():.4f} train_acc: {acc_train:.4f} val_acc: {acc_val:.4f} test_acc: {acc_test:.4f}")

    # return best_model
    return model


def evaluate(model, graph, feature, idx):
    model.eval()
    output =  model(graph, feature)
    labels = graph.ndata['label']
    acc_val = accuracy(output[idx], labels[idx])
    return acc_val

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_dataset(dataset_name)
    args.num_features = num_features

    acc_list = []
    # estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        model = BuildModel(args.backbone, num_features, args.n_hid, num_classes, args.n_layers, args.activation, args.norm, args.drop, args.residual_type).build()

        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            # logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]
        logger = None
        if not load_model:
            model = train(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            # logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            # logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        model = model.to(device)
        model.eval()
        test_mask = graph.ndata['test_mask']
        final_acc = evaluate(model, graph, x, test_mask)

        # final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        # estp_acc_list.append(estp_acc)



    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    # estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    # print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)