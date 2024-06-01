# PSNR-GNN

Develop a node-adaptive residual module for deep graph neural network via posteriori sampling that alleviates over-smoothing.

![PSNR_Arc & Exp](fig/PSNR_Arc & Exp.svg)

# Installation

To install the required dependencies for PSNR-GNN, you can use the provided `requirement.txt` file. 

``` 
bash pip install -r requirements.txt
```

The `requirements.txt` file includes the following dependencies:
```
    dgl==2.1.0+cu118
    numpy==1.24.1
    ogb==1.3.6
    pandas==2.2.2
    PyYAML==6.0.1
    scikit_learn==1.4.2
    torch==2.2.1+cu118
    tqdm==4.65.0
```

# Usage


## To run Classical Node Classification Task

```
python run_transductive.py
```

## To run Classical Node Classification with Missing Vector Task

```
python run_transductive_mv.py
```