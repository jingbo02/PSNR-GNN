import wandb
from main_transductive import main
# Press the green button in the gutter to run the script.

def build_wandb_args():
    args = {}
    args["method"] = "grid"
    metric = {
        'name': 'avg_test',
        'goal': 'maximize'   
        }
    args['metric'] = metric
    args['parameters'] = {}
    args['parameters'].update({
        # Hyperparameters
        'backbone': {'value': 'gcn'},
        'n_layers': {'value': 2},
        'residual_type': {'value': 'snr'},
        'max_epoch': {'value': 500},
        'randn_init': {'value': False}, # Initialization for parameter of SNRModule
        'activation': {'value': 'elu'},
        'seeds': {'value': [2024]}, #TODO 固定seed
        'dataset': {'value': 'cora'},
        'split_dataset': {'value': False},
        'pre_split_path': {'value': './datasets/split_data'},                  
        'loda_split': {'value': False}, #TODO delete save_split
        'num_split': {'value': 5},
        'device': {'value': 1},
        'num_heads': {'value': 4}, # number of hidden attention heads
        'optimizer': {'value': 'adam'},
        'use_cfg': {'value': False}, # if load best config
        'logging': {'value': False},
        'log_path': {'value': './logging_data'},

        # Hyperparameters Under Optimization
        "n_hid": {'values': [64, 128, 256]},
        'lr': {'values': [1e-3, 1e-4]}, # learning rate
        'weight_decay': {'values': [5e-4]},
        'drop': {'value': [0.1,0.1]},
        'norm': {'value': []},
        })

    return args

if __name__ == "__main__":
    project_name = 'test'
    wandb.login(
        host='https://api.wandb.ai',
        key='aa45b3ed8e0b4f2ad798b9e7fd687c3be8d8cf50',
    )
    sweep_config = build_wandb_args()
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, main, count=1)


