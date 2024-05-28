import pandas as pd
from main_transductive_mv import main
import argparse
import itertools

sweep_config = {
    'backbone': ['gcn', 'gat'],
    'max_epoch': [1000],
    'activation': ['elu'],
    'seeds': [[0, 1, 2, 3, 4]], 
    'device': [0],
    'num_heads': [1], # number of hidden attention heads
    'optimizer': ['adam'],
    'n_hid': [32],
    'drop': [[0.5, 0]],
    'norm': [[]],
    'layer_emb': [True],
    'if_mv': [True],
    'if_early_stop': [False],
    
    # Encoder
    'n_layers':[2, 4, 6, 8, 10, 15, 20, 30],
    'lr':[1e-2, 1e-3],
    'residual_type': ['snr'],
    'weight_decay': [5e-4],
    'dataset': ['cora','citeseer','pubmed'],
    'coef_encoder': ['gat','mlp','gcn'],
}
# 
project_name = 'mv/GAT_32hid_noearlystop_drop_wd_snr_encoder'

results_table = pd.DataFrame(columns = list(sweep_config.keys()) + ['final_acc', 'final_acc_std', 'val_acc', 'val_acc_std'])
parser = argparse.ArgumentParser()
args = parser.parse_args()

print(project_name)

# i = 0
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