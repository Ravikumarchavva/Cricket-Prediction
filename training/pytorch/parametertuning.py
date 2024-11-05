import os
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F  # Import F module
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import wandb  # Import wandb
from wandb import config  # Import config from wandb


# Load the data using polars
directory = r'D:\github\Cricket-Prediction\data\filteredData'
balltoball = pl.read_csv(os.path.join(directory, 'balltoball.csv'))
teamStats = pl.read_csv(os.path.join(directory, 'team12Stats.csv'))
playersStats = pl.read_csv(os.path.join(directory, 'playersStats.csv'))

# Preprocess the data
def partition_data(df, group_keys):
    partitions = df.partition_by(group_keys)
    partition_dict = {}
    for partition in partitions:
        match_id = partition[group_keys[0]][0]
        flip = partition[group_keys[1]][0]
        key = (match_id, flip)
        partition_dict[key] = partition.drop(group_keys).to_numpy()
    return partition_dict

# Update group_keys to ['match_id', 'flip', 'innings']
team_stats_partitions = partition_data(teamStats, ['match_id', 'flip', 'innings'])
player_stats_partitions = partition_data(playersStats, ['match_id', 'flip', 'innings'])
ball_stats_partitions = partition_data(balltoball, ['match_id', 'flip', 'innings'])

# Split match IDs into training and validation sets
match_ids = list(set([key[0] for key in team_stats_partitions.keys()]))
train_matches, val_matches = train_test_split(match_ids, test_size=0.2, random_state=42)

# Function to prepare data with overs limit
def prepare_data(matches, is_validation=False):
    team_stats_list = []
    player_stats_list = []
    ball_stats_list = []

    for match_id in matches:
        for flip in [0, 1]:  # Team perspectives
            for innings in [1, 2]:  # Innings numbers
                key = (match_id, flip, innings)
                if key in team_stats_partitions and key in player_stats_partitions and key in ball_stats_partitions:
                    team_stats = team_stats_partitions[key]
                    player_stats = player_stats_partitions[key]
                    ball_stats = ball_stats_partitions[key]

                    # Determine max overs
                    if is_validation and innings == 2:
                        max_overs = 16  # Limit second innings to 16 overs in validation data
                    else:
                        max_overs = 20  # Up to 20 overs

                    total_overs = ball_stats.shape[0] // 6  # Assuming 6 balls per over
                    max_overs = min(total_overs, max_overs)
                    end_idx = max_overs * 6

                    # Append the data
                    team_stats_list.append(team_stats)
                    player_stats_list.append(player_stats)
                    ball_stats_list.append(ball_stats[:end_idx])

    return team_stats_list, player_stats_list, ball_stats_list

# Prepare training data
train_team_stats, train_player_stats, train_ball_stats = prepare_data(train_matches, is_validation=False)

# Prepare validation data
val_team_stats, val_player_stats, val_ball_stats = prepare_data(val_matches, is_validation=True)

# Create a custom Dataset
class CricketDataset(Dataset):
    def __init__(self, team_stats_list, player_stats_list, ball_stats_list):
        self.team_stats_list = team_stats_list
        self.player_stats_list = player_stats_list
        self.ball_stats_list = ball_stats_list

    def __len__(self):
        return len(self.team_stats_list)

    def __getitem__(self, idx):
        team_input = torch.tensor(self.team_stats_list[idx], dtype=torch.float32)
        team_input = team_input.squeeze()  # Remove extra dimensions
        player_input = torch.tensor(self.player_stats_list[idx], dtype=torch.float32)
        ball_stats = torch.tensor(self.ball_stats_list[idx], dtype=torch.float32)
        # Assuming the last column is the label
        ball_input = ball_stats[:, :-1]
        label = ball_stats[0, -1]
        return team_input, player_input, ball_input, label

# Define a collate function to handle variable-length sequences
def collate_fn(batch):
    team_inputs = []
    player_inputs = []
    ball_inputs = []
    labels = []
    ball_lengths = []

    for team_input, player_input, ball_input, label in batch:
        team_inputs.append(team_input)
        player_inputs.append(player_input)
        ball_inputs.append(ball_input)
        labels.append(label)
        ball_lengths.append(ball_input.shape[0])

    # Pad ball_inputs to the maximum sequence length in the batch
    max_seq_len = max(ball_lengths)
    padded_ball_inputs = torch.zeros(len(ball_inputs), max_seq_len, ball_inputs[0].shape[1])
    for i, ball_input in enumerate(ball_inputs):
        seq_len = ball_input.shape[0]
        padded_ball_inputs[i, :seq_len, :] = ball_input

    team_inputs = torch.stack(team_inputs)
    player_inputs = torch.stack(player_inputs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return team_inputs, player_inputs, padded_ball_inputs, labels, ball_lengths

# Create the training and validation datasets and dataloaders
train_dataset = CricketDataset(train_team_stats, train_player_stats, train_ball_stats)
val_dataset = CricketDataset(val_team_stats, val_player_stats, val_ball_stats)

# Define the models
class TeamStatsModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(TeamStatsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 16),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

class PlayerStatsModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(PlayerStatsModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size * 2 * ((input_size - 4) // 4), 16)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch, channels, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x

class BallToBallModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_heads, num_layers, dropout):
        super(BallToBallModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 16)

    def forward(self, x, lengths):
        x = self.embedding(x)
        # Create a mask to ignore padding tokens
        mask = (torch.arange(x.size(1))[None, :] < lengths[:, None]).to(x.device)
        mask = ~mask
        x = self.transformer(x, x, src_key_padding_mask=mask, tgt_key_padding_mask=mask)
        x = x.mean(dim=1)  # Average pooling over the sequence length
        x = F.relu(self.fc(x))
        return x

class CombinedModel(nn.Module):
    def __init__(self, team_input_size, player_input_size, ball_input_dim, hidden_size, num_heads, num_layers, dropout):
        super(CombinedModel, self).__init__()
        self.team_model = TeamStatsModel(team_input_size, hidden_size, dropout)
        self.player_model = PlayerStatsModel(player_input_size, hidden_size, dropout)
        self.ball_model = BallToBallModel(ball_input_dim, hidden_size, num_heads, num_layers, dropout)
        self.fc = nn.Sequential(
            nn.Linear(16+16+16, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, team_input, player_input, ball_input, ball_lengths):
        team_output = self.team_model(team_input)
        player_output = self.player_model(player_input)
        ball_output = self.ball_model(ball_input, ball_lengths)
        combined = torch.cat((team_output, player_output, ball_output), dim=1)
        output = self.fc(combined)
        return output.squeeze()

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
first_key = next(iter(team_stats_partitions))
team_input_size = team_stats_partitions[first_key].shape[1]
player_input_size = player_stats_partitions[first_key].shape[1]
ball_input_dim = ball_stats_partitions[first_key].shape[1] - 1  # Exclude label

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def main():
    # Initialize wandb with project name and config
    wandb.init(project="cricket-prediction")
    config = wandb.config

    # Log the hyperparameters
    print(f"Running with config: {config}")

    # Load the data
    # (Data loading code is already executed above)

    # Update dataloaders to use hyperparameters from wandb.config
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize the model with hyperparameters from wandb.config
    model = CombinedModel(team_input_size, player_input_size, ball_input_dim, config.hidden_size, config.num_heads, config.num_layers, config.dropout).to(device)

    # Define optimizer and scheduler with hyperparameters from wandb.config
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    criterion = nn.BCELoss()

    # Implement early stopping with patience from config
    best_loss = np.inf
    trigger_times = 0

    # Training loop
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for team_input, player_input, ball_input, labels, ball_lengths in progress_bar:
            team_input, player_input, ball_input, labels = team_input.to(device), player_input.to(device), ball_input.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(team_input, player_input, ball_input, ball_lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item())
        avg_train_loss = running_loss / len(train_dataloader)
        train_accuracy = correct_predictions / total_predictions
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    
        # Log training metrics to wandb
        wandb.log({"Train Loss": avg_train_loss, "Train Accuracy": train_accuracy})
    
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        with torch.no_grad():
            for team_input, player_input, ball_input, labels, ball_lengths in val_dataloader:
                team_input, player_input, ball_input, labels = team_input.to(device), player_input.to(device), ball_input.to(device), labels.to(device)  # Move data to GPU
                outputs = model(team_input, player_input, ball_input, ball_lengths)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                val_correct_predictions += (predictions == labels).sum().item()
                val_total_predictions += labels.size(0)
        avg_val_loss = val_running_loss / len(val_dataloader)
        val_accuracy = val_correct_predictions / val_total_predictions
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
        # Log validation metrics to wandb
        wandb.log({"Val Loss": avg_val_loss, "Val Accuracy": val_accuracy})
    
        # Early stopping logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            trigger_times = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            wandb.save('best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= config.patience:
                print('Early stopping!')
                break
    
        # Step the scheduler
        scheduler.step()
    
    # Save the final model
    torch.save(model.state_dict(), 't20i.pth')
    wandb.save('t20i.pth')
    
    # Finish the wandb run
    wandb.finish()

if __name__ == '__main__':
    sweep_config = {
        'method': 'bayes',  # Change to Bayesian search
        'metric': {
            'name': 'Val Loss',
            'goal': 'minimize'
        },
        'parameters': {
            'batch_size': {
                'values': [16, 32, 64]
            },
            'learning_rate': {
                'values': [1e-3, 1e-4, 1e-2]
            },
            'dropout': {
                'values': [0.3, 0.5, 0.7]
            },
            'hidden_size': {
                'values': [64, 128, 256]
            },
            'weight_decay': {
                'values': [0, 1e-5, 1e-4]
            },
            'scheduler_step_size': {
                'values': [5, 10]
            },
            'scheduler_gamma': {
                'values': [0.1, 0.5]
            },
            'patience': {
                'values': [5, 10]
            },
            'num_epochs': {
                'value': 50
            },
            'num_heads': {
                'values': [2, 4, 8]
            },
            'num_layers': {
                'values': [2, 4, 6]
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5,
        },
        'count': 20  # Limit the number of runs to 20
    }

    sweep_id = wandb.sweep(sweep_config, project="cricket-prediction")
    wandb.agent(sweep_id, function=main, count=20)  # Run the sweep agent with 20 runs
