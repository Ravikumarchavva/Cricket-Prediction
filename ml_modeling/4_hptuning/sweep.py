import wandb
import yaml
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_utils import set_seed
from data_utils import load_datasets
import train

# Load the sweep configuration
with open(os.path.join(os.path.dirname(__file__), 'hpconfig.yaml')) as config_file:
    sweep_config = yaml.safe_load(config_file)

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="T20I-CRICKET-WINNER-PREDICTION")

# Load datasets once
train_dataset, val_dataset, test_dataset = load_datasets()

def train_sweep():
    set_seed()
    wandb.init(reinit=True)  # Allow multiple initializations
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics, all_labels, all_predictions, all_probs = train.train(config, train_dataset, val_dataset, test_dataset, device)
    overall_metrics = metrics["overall_metrics"]
    stage_metrics = metrics["stage_metrics"]
    for stage, metrics in stage_metrics.items():
        wandb.log({stage: metrics})

    wandb.log({"overall_accuracy": overall_metrics["accuracy"]})
    wandb.finish()

# Start the sweep agent
wandb.agent(sweep_id, function=train_sweep, count=50)