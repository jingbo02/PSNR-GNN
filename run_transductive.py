import pandas as pd
from main_transductive import main
import argparse
import itertools

sweep_config = {
    'backbone': ['gcn', 'gat'],
    'max_epoch': [500],
    'activation': ['elu'],
    'seed': [42], 
    'pre_split_path': ['split_datasets/'],      # Path of data splits         
    'loda_split':[False],      # If load prepared splits. If False, then it will randomly split the datasets.
    'device': [0],
    'num_heads': [3], # Number of hidden attention heads for the backbone GAT
    'optimizer': ['adam'],
    'n_hid': [64],
    'norm': [[]],  #TODO Type of Normalization: 
    'drop':[[0.5, 0]], # [drop_out_ratio, drop_edge_ratio]
    'layer_emb': [True], # If add layer embedding for PSNR
    'if_early_stop': [True],
    'split_type': ['semi'], # Type of Datasets splits: 'semi' for classical, 'hetero' for train:val:test = 6:2:2, 'full' for train:val:test = 2:2:6
    'n_layers': [2, 4, 8, 16, 32, 64],
    'lr': [1e-2, 1e-3], # learning rate
    'residual_type': ['psnr', 'None'],
    'weight_decay': [5e-4],
    'dataset': ['cora', 'citeseer', 'pubmed', 'coauther_cs', 'coauther_phy', 'amazon_photo'],
    'coef_encoder': ['sage'],  # Type of the encoder of PSNR
}

project_name = 'baseline'

results_table = pd.DataFrame(columns = list(sweep_config.keys()) + ['final_acc', 'final_acc_std', 'val_acc', 'val_acc_std'])
parser = argparse.ArgumentParser()
args = parser.parse_args()

print(project_name)

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
