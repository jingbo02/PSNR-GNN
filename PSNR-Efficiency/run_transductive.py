







import pandas as pd
from main_transductive import main
import argparse
import itertools

sweep_config = {
    'backbone':['gcn'],
    'max_epoch': [500],
    'activation': ['elu'],
    'seed': [42], 
    'pre_split_path': ['split_datasets/'],                  
    'loda_split':[True], 
    'device': [0],
    'num_heads': [2], # number of hidden attention heads
    'optimizer': ['adam'],
    'n_hid': [64],
    'norm': [[]],
    'drop':[[0.5, 0]],
    'layer_emb': [True],
    'if_mv': [False],
    'if_early_stop': [True],
    'if_var': [True],
    'if_adjust_lr': [False],

    # Hyperparameters Under Optimization

    # Encoder
    'n_layers': [4],
    'lr':[1e-2],
    'residual_type': [
        'none',
        # 'dense',
        # 'snr', 
        # 'res', 
        # 'jk'
    ],
    'weight_decay': [0],
    'dataset': ['ogbn-arxiv'],
    'coef_encoder': ['sage'],
}
# 
project_name = 'large_graph_time'

results_table = pd.DataFrame(columns = list(sweep_config.keys()) + ['final_acc', 'final_acc_std', 'val_acc', 'val_acc_std'])
parser = argparse.ArgumentParser()
args = parser.parse_args()

print(project_name)

i = 0
for values in itertools.product(*sweep_config.values()):

    params = dict(zip(sweep_config.keys(), values))
        
    for key, value in params.items():
        setattr(args, key, value)
        
    if args.if_adjust_lr:
        if args.n_layers >= 15:
            args.lr = 0.001
    
    print(args.n_layers, args.residual_type, args.dataset)
    final_acc, final_acc_std, val_acc, val_acc_std = main(args)
    
    result = args.__dict__.copy()
    result.update({
        'final_acc': final_acc, 
        'final_acc_std': final_acc_std, 
        'val_acc': val_acc, 
        'val_acc_std': val_acc_std
    })
    # results_table = results_table._append(result, ignore_index = True)
    # results_table.to_csv(project_name + '.csv', index=True)
