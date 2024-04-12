import wandb
from main_transductive import main

def build_wandb_args():
    args = {}
    args["method"] = "grid"
    metric = {
        'name': 'final_acc',
        'goal': 'maximize'   
        }
    args['metric'] = metric
    args['parameters'] = {}
    args['parameters'].update({
        # Hyperparameters
        'backbone': {'value': 'gcn'},
        'residual_type': {'value': 'init_res'},
        'max_epoch': {'value': 500},
        'randn_init': {'value': False}, # Initialization for parameter of SNRModule
        'activation': {'value': 'elu'},
        'seed': {'value': 2024}, 
        'dataset': {'value': 'cora'},
        'pre_split_path': {'value': 'datasets/split_data'},                  
        'loda_split': {'value': True}, 
        'device': {'value': 0},
        'num_heads': {'value': 3}, # number of hidden attention heads
        'optimizer': {'value': 'adam'},
        "n_hid": {'value': 64},
        'drop': {'value': [0.5, 0]},
        'norm': {'value': []},

        # Hyperparameters Under Optimization
        'n_layers': {'values': [2,4,8,16,32,64]},
        'lr': {'values': [1e-2, 1e-3]}, # learning rate
        'weight_decay': {'values': [1e-3,5e-4,1e-4]},
    })

    return args

if __name__ == "__main__":
    project_name = 'Cora'
    wandb.login(
        host='https://api.wandb.ai',
        key='79bd072f9b61735f983f9da5acbd2b78383c4268',
    )
    sweep_config = build_wandb_args()
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, main, count=36)


