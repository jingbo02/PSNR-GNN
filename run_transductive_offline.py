import pandas as pd
from main_transductive import main
import argparse
import wandb
import itertools

sweep_config = {
    'backbone': ['gcn'],
    'max_epoch': [500],
    'activation': ['elu'],
    'seed': [42], 
    'pre_split_path': ['split_datasets/'],                  
    'loda_split':[True], 
    'device': [7],
    'num_heads': [3], # number of hidden attention heads
    'optimizer': ['adam'],
    "n_hid": [128],
    'drop': [[0.7, 0]],
    'norm': [[]],
    'layer_emb': [True],

    # Hyperparameters Under Optimization

    # Encoder
    'n_layers': [2, 4, 8, 16, 32, 64],
    'lr': [1e-2, 1e-3], # learning rate
    'residual_type': ['snr'],
    'weight_decay': [5e-4],
    'dataset': ['cora','citeseer','pubmed','coauther_cs','coauther_phy','amazon_photo'],
    'coef_encoder': ['sage','mlp','gat','gcn'],

}

project_name = 'dropout_search_7'

results_table = pd.DataFrame(columns = list(sweep_config.keys()) + ['final_acc', 'final_acc_std', 'val_acc', 'val_acc_std'])
# import pdb ; pdb.set_trace()
parser = argparse.ArgumentParser()
args = parser.parse_args()

# 网格化搜参数
i = 0
for values in itertools.product(*sweep_config.values()):

    params = dict(zip(sweep_config.keys(), values))
    for key, value in params.items():
        setattr(args, key, value)

    final_acc, final_acc_std, val_acc, val_acc_std = main(args)
    result = args.__dict__.copy()
    result.update({
        'final_acc': final_acc, 
        'final_acc_std': final_acc_std, 
        'val_acc': val_acc, 
        'val_acc_std': val_acc_std
    })
    results_table = results_table._append(result, ignore_index = True)
    results_table.to_csv(project_name + '.csv', index=True)
    # import pdb;  pdb.set_trace()