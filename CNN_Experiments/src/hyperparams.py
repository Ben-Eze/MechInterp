import json


def read(file_path):
    """Loads hyperparameters from a .json file."""
    with open(file_path, 'r') as f:
        hyperparams = json.load(f)

    return hyperparams['data_hyperparams'], \
           hyperparams['model_hyperparams'], \
           hyperparams['train_hyperparams']