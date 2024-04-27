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
        'max_epoch': {'value': 500},
        'activation': {'value': 'elu'},
        'seed': {'value': 42}, 
        'pre_split_path': {'value': 'split_datasets/'},                  
        'loda_split': {'value': True}, 
        'device': {'value': 6},
        'num_heads': {'value': 3}, # number of hidden attention heads
        'optimizer': {'value': 'adam'},
        "n_hid": {'value': 128},
        'drop': {'value': [0.3, 0]},
        'norm': {'value': []},
        'layer_emb': { 'value': True},
 
        # Hyperparameters Under Optimization
        

        # Encoder
        'n_layers': {'values': [2, 4, 8, 16, 32, 64]},
        'lr': {'values': [1e-2, 1e-3]}, # learning rate
        'residual_type': {'values': ['snr']},
        'weight_decay': {'values': [5e-4]},
        'dataset': {'values': ['cora','citeseer','pubmed','coauther_cs','coauther_phy','amazon_photo']},
        'coef_encoder': {'values': ['sage','mlp','gat','gcn']},
    })

    return args

if __name__ == "__main__":
    project_name = 'dropout_search'
    wandb.login(
        host = 'https://api.wandb.ai',
        key = '79bd072f9b61735f983f9da5acbd2b78383c4268',
    )
    # wandb.login(
    # host='https://api.wandb.ai',
    # key='aa45b3ed8e0b4f2ad798b9e7fd687c3be8d8cf50',
    # )
    sweep_config = build_wandb_args()
    sweep_id = wandb.sweep(sweep_config, project = project_name)
    wandb.agent(sweep_id, main, count = 288)


