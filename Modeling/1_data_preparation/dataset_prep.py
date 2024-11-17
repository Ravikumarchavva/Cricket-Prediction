import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.join(os.getcwd(),".."))
from model_utils import CricketDataset, collate_fn_with_padding

import polars as pl
# import data
def load_data():
    balltoball = pl.read_csv(os.path.join(os.path.join( '..',"data", "filtered_data" , "balltoball.csv")))
    team_stats = pl.read_csv(os.path.join(os.path.join( '..',"data", "filtered_data" , "team12_stats.csv")))
    players_stats = pl.read_csv(os.path.join(os.path.join( '..',"data", "filtered_data" , "players_stats.csv")))
    return balltoball, team_stats, players_stats
balltoball,team_stats,players_stats = load_data()

def partition_data_with_keys(df, group_keys):
    partitions = df.partition_by(group_keys)
    keys = [tuple(partition.select(group_keys).unique().to_numpy()[0]) for partition in partitions]
    partitions = [partition.drop(group_keys).to_numpy() for partition in partitions]
    return keys, partitions

# Use the updated partition_data_with_keys function
balltoball_keys, balltoball_partitions = partition_data_with_keys(balltoball, ["match_id", "flip"])
team_stats_keys, team_stats_partitions = partition_data_with_keys(team_stats, ["match_id", "flip"])
players_stats_keys, players_stats_partitions = partition_data_with_keys(players_stats, ["match_id", "flip"])

# Align the partitions using common keys
common_keys = set(balltoball_keys) & set(team_stats_keys) & set(players_stats_keys)

balltoball_dict = dict(zip(balltoball_keys, balltoball_partitions))
team_stats_dict = dict(zip(team_stats_keys, team_stats_partitions))
players_stats_dict = dict(zip(players_stats_keys, players_stats_partitions))

aligned_balltoball_partitions = []
aligned_team_stats_partitions = []
aligned_players_stats_partitions = []
labels = []

for key in common_keys:
    balltoball_partition = balltoball_dict[key]
    team_stats_partition = team_stats_dict[key]
    players_stats_partition = players_stats_dict[key]

    label = balltoball_partition[:, -1][0]
    aligned_balltoball_partitions.append(balltoball_partition[:, :-1])
    aligned_team_stats_partitions.append(team_stats_partition)
    aligned_players_stats_partitions.append(players_stats_partition)
    labels.append(label)

import numpy as np
labels = np.array(labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Compute class weights
from torch import tensor
pos_count = np.sum(labels == 1)
neg_count = np.sum(labels == 0)
if pos_count > 0 and neg_count > 0:
    pos_weight = neg_count / pos_count
    pos_weight_tensor = tensor([pos_weight]).to(device)
    # Update the loss function with pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
else:
    # Fallback if no imbalance
    criterion = nn.BCEWithLogitsLoss()

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# Update the dataset initialization to include windowing parameters
dataset = CricketDataset(
    aligned_team_stats_partitions,
    aligned_players_stats_partitions,
    aligned_balltoball_partitions,
    labels
)

train_indices, temp_indices = train_test_split(
    list(range(len(dataset))), stratify=labels, test_size=0.3, random_state=42)

val_indices, test_indices = train_test_split(
    temp_indices, stratify=[labels[i] for i in temp_indices], test_size=0.5, random_state=42)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_with_padding)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_with_padding)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_with_padding)

for team_input, player_input, ball_input, labels, mask in train_dataloader:
    print(f"Team input shape: {team_input.shape}")  # [batch_size, team_feature_dim]
    print(f"Player input shape: {player_input.shape}")  # [batch_size, player_feature_dim]
    print(f"Padded ball input shape: {ball_input.shape}")  # [batch_size, max_seq_len, ball_feature_dim]
    print(f"Mask shape: {mask.shape}")  # [batch_size, max_seq_len]
    print(f"Labels shape: {labels.shape}")  # [batch_size]
    break


import pickle
with open(os.path.join( '..',"data", "pytorch_data" , "train_dataloader.pkl"), "wb") as f:
    pickle.dump(train_dataloader, f)

with open(os.path.join( '..',"data", "pytorch_data" , "val_dataloader.pkl"), "wb") as f:
    pickle.dump(val_dataloader, f)

with open(os.path.join( '..',"data", "pytorch_data" , "test_dataloader.pkl"), "wb") as f:
    pickle.dump(test_dataloader, f)