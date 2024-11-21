import wandb
import yaml
import os
import subprocess

# Load the sweep configuration
with open(os.path.join(os.path.dirname(__file__), 'hpconfig.yaml')) as config_file:
    sweep_config = yaml.safe_load(config_file)

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="T20I")

# Run the sweep agent
wandb.agent(sweep_id, function=lambda: subprocess.call(["python", "sweep.py"]), count=50)