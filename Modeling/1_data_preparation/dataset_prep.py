import os
import sys
sys.path.append(os.path.join(os.getcwd(),".."))
from model_utils import CricketDataset, collate_fn_with_packing, partition_data_with_keys

import polars as pl
# import data
def load_data():
    balltoball = pl.read_csv(os.path.join(os.path.join( '..',"data", "filtered_data" , "balltoball.csv")))
    team_stats = pl.read_csv(os.path.join(os.path.join( '..',"data", "filtered_data" , "team12_stats.csv")))
    players_stats = pl.read_csv(os.path.join(os.path.join( '..',"data", "filtered_data" , "players_stats.csv")))
    return balltoball, team_stats, players_stats
balltoball,team_stats,players_stats = load_data()


# Use the updated partition_data_with_keys function
balltoball_keys, balltoball_partitions = partition_data_with_keys(balltoball, ["match_id"])
team_stats_keys, team_stats_partitions = partition_data_with_keys(team_stats, ["match_id"])
players_stats_keys, players_stats_partitions = partition_data_with_keys(players_stats, ["match_id"])

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

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_with_packing)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_with_packing)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_with_packing)


import pickle
with open(os.path.join( '..',"data", "pytorch_data" , "train_dataloader.pkl"), "wb") as f:
    pickle.dump(train_dataloader, f)

with open(os.path.join( '..',"data", "pytorch_data" , "val_dataloader.pkl"), "wb") as f:
    pickle.dump(val_dataloader, f)

with open(os.path.join( '..',"data", "pytorch_data" , "test_dataloader.pkl"), "wb") as f:
    pickle.dump(test_dataloader, f)
