import os
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F  # Import F module
from sklearn.model_selection import train_test_split

# Load the data using polars
directory = r'D:\github\localt20\data\filteredData'
balltoball = pl.read_csv(os.path.join(directory, 'balltoball.csv'))
teamStats = pl.read_csv(os.path.join(directory, 'team12Stats.csv'))
playersStats = pl.read_csv(os.path.join(directory, 'playersStats.csv'))

# Preprocess the data
def partition_data(df, group_keys):
    partitions = df.partition_by(group_keys)
    partition_list = [partition.drop(group_keys).to_numpy() for partition in partitions]
    return partition_list

team_stats_partitions = partition_data(teamStats, ['match_id', 'flip'])
player_stats_partitions = partition_data(playersStats, ['match_id', 'flip'])
ball_stats_partitions = partition_data(balltoball, ['match_id', 'flip'])

# Augment the data by creating new samples with different combinations of overs
def augment_data(team_stats_list, player_stats_list, ball_stats_list, over_segments=[6, 10, 15]):
    augmented_team_stats = []
    augmented_player_stats = []
    augmented_ball_stats = []
    
    for team_stats, player_stats, ball_stats in zip(team_stats_list, player_stats_list, ball_stats_list):
        total_overs = ball_stats.shape[0] // 6  # Assuming 6 balls per over
        for segment in over_segments:
            if total_overs >= segment:
                end_idx = segment * 6
                augmented_team_stats.append(team_stats)
                augmented_player_stats.append(player_stats)
                augmented_ball_stats.append(ball_stats[:end_idx])
    
    return augmented_team_stats, augmented_player_stats, augmented_ball_stats

augmented_team_stats, augmented_player_stats, augmented_ball_stats = augment_data(
    team_stats_partitions, player_stats_partitions, ball_stats_partitions)

# Split the dataset into training and validation sets
_, val_team_stats, _, val_player_stats, _, val_ball_stats = train_test_split(
    augmented_team_stats, augmented_player_stats, augmented_ball_stats, test_size=0.2, random_state=42)

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

# Create the validation dataset and dataloader
val_dataset = CricketDataset(val_team_stats, val_player_stats, val_ball_stats)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Define the models
class TeamStatsModel(nn.Module):
    def __init__(self, input_size):
        super(TeamStatsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

class PlayerStatsModel(nn.Module):
    def __init__(self, input_size):
        super(PlayerStatsModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * ((input_size - 4) // 4), 16)

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
    def __init__(self, input_dim):
        super(BallToBallModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 16)

    def forward(self, x, lengths):
        # Pack the sequences
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output_packed, (hn, cn) = self.lstm(x_packed)
        # Concatenate the final forward and backward hidden states
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        x = self.dropout(hn)
        x = F.relu(self.fc(x))
        return x

class CombinedModel(nn.Module):
    def __init__(self, team_input_size, player_input_size, ball_input_dim):
        super(CombinedModel, self).__init__()
        self.team_model = TeamStatsModel(team_input_size)
        self.player_model = PlayerStatsModel(player_input_size)
        self.ball_model = BallToBallModel(ball_input_dim)
        self.fc = nn.Sequential(
            nn.Linear(16+16+16, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
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
team_input_size = team_stats_partitions[0].shape[1]
player_input_size = player_stats_partitions[0].shape[1]
ball_input_dim = ball_stats_partitions[0].shape[1] - 1  # Exclude label

model = CombinedModel(team_input_size, player_input_size, ball_input_dim).to(device)  # Move model to GPU

# Load the saved model
model.load_state_dict(torch.load('t20i.pth', weights_only=True))
model.eval()

# Take a single sample from the validation set and make a prediction
val_iter = iter(val_dataloader)
team_input, player_input, ball_input, label, ball_lengths = next(val_iter)

team_input, player_input, ball_input, label = team_input.to(device), player_input.to(device), ball_input.to(device), label.to(device)  # Move data to GPU

with torch.no_grad():
    output = model(team_input, player_input, ball_input, ball_lengths)
    prediction = (output > 0.5).float()

print(f'Prediction: {prediction.cpu().numpy()}')
print(f'Actual Label: {label.cpu().numpy()}')
