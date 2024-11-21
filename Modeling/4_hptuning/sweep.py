import wandb
import yaml
import os
import subprocess

# Load the sweep configuration
with open(os.path.join(os.path.dirname(__file__), 'hpconfig.yaml')) as config_file:
    sweep_config = yaml.safe_load(config_file)

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="T20I-augmented-sweep1")


# Define the function to run the training script
def train():
    # Execute the training script
    subprocess.run(["python", "../3_augumented_training/ag_train.py"])

# Start the sweep agent
wandb.agent(sweep_id, function=train, count=50)