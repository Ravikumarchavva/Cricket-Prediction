import os
import sys
import pickle
import torch
import wandb
import numpy as np

# Initialize Weights & Biases
wandb.init(project="T20I-CRICKET-WINNER-PREDICTION", job_type="inference")

# Get the best model artifact
artifact = wandb.use_artifact('ravikumarchavva-org/T20I-CRICKET-WINNER-PREDICTION/best_model:latest', type='model')
artifact_dir = artifact.download()
model_path = os.path.join(artifact_dir, 'best_model.pth')

# Load model hyperparameters from artifact metadata
config = artifact.metadata.get('config', {})
fc1_input_size = artifact.metadata.get('fc1_input_size', None)
if not config or fc1_input_size is None:
    raise ValueError("Model hyperparameters not found in artifact metadata.")

# Add parent directory to sys.path to import model_utils
sys.path.append(os.path.join(os.getcwd(), '..'))
# Add the path to the training directory to sys.path
sys.path.append(os.path.join(os.getcwd(), '..'))
# Import model_utils from the training directory
from model_utils import (
    EncoderDecoderModel, collate_fn_with_padding, extract_data,
    augment_match_data, CricketDataset
)
from torch.utils.data import DataLoader

# Load the test dataset
test_dataset = pickle.load(open(os.path.join(
    os.getcwd(), '..', "data", "pytorch_data", 'test_dataset.pkl'), 'rb'))

# Extract and augment data
test_team_data, test_player_data, test_ball_data, test_labels = extract_data(test_dataset)
test_ball_data, test_team_data, test_player_data, test_labels = augment_match_data(
    test_ball_data, test_team_data, test_player_data, test_labels
)

# Create PyTorch Dataset and DataLoader
test_dataset = CricketDataset(
    test_team_data, test_player_data, test_ball_data, test_labels)
test_dataloader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_with_padding)

# Initialize the model with the same hyperparameters used during training
model = EncoderDecoderModel(
    team_input_size=test_team_data[0].shape[0],
    player_input_channels=1,
    player_fc1_input_size=fc1_input_size,
    ball_input_size=test_ball_data[0].shape[1],
    hidden_size=config.get('hidden_size', 128),
    num_layers=config.get('num_layers', 2),
    num_classes=1,
    dropout=config.get('dropout', 0.5)
).to(device)

# Load the model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Run predictions on test data
all_predictions = []
with torch.no_grad():
    for team, player, ball, labels in test_dataloader:
        team, player, ball = team.to(device), player.to(device), ball.to(device)
        outputs = model(team, player, ball)
        predicted = (outputs > 0.5).float()
        all_predictions.extend(predicted.cpu().numpy())

# Output predictions
print("Predictions on test data:")
print(all_predictions)