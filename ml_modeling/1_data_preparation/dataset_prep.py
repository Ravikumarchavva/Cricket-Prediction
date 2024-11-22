import os
import sys
import numpy as np
import polars as pl
import pickle
from typing import Tuple

sys.path.append(os.path.join(os.getcwd(), ".."))
from model_utils import CricketDataset, partition_data_with_keys

# Step 1: Load Data
def load_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Loads the ball-by-ball data, team statistics, and player statistics.

    Returns:
        Tuple containing ball-by-ball DataFrame, team statistics DataFrame, and player statistics DataFrame.
    """
    balltoball = pl.read_csv(os.path.join(os.path.join('..', "data", "filtered_data", "balltoball.csv")))
    team_stats = pl.read_csv(os.path.join(os.path.join('..', "data", "filtered_data", "team12_stats.csv")))
    players_stats = pl.read_csv(os.path.join(os.path.join('..', "data", "filtered_data", "players_stats.csv")))
    return balltoball, team_stats, players_stats

balltoball, team_stats, players_stats = load_data()

# Step 2: Partition Data

balltoball_keys, balltoball_partitions = partition_data_with_keys(balltoball, ["match_id"])
team_stats_keys, team_stats_partitions = partition_data_with_keys(team_stats, ["match_id"])
players_stats_keys, players_stats_partitions = partition_data_with_keys(players_stats, ["match_id"])

# Step 3: Align Partitions
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
    aligned_balltoball_partitions.append(balltoball_partition[:-1, :-1])  # remove the last row and column
    aligned_team_stats_partitions.append(team_stats_partition)
    aligned_players_stats_partitions.append(players_stats_partition)
    labels.append(label)

labels = np.array(labels)

# Step 4: Prepare Data for Training
team_data = [team.to_numpy() if isinstance(team, pl.DataFrame) else team for team in aligned_team_stats_partitions]
player_data = [players.to_numpy() if isinstance(players, pl.DataFrame) else players for players in aligned_players_stats_partitions]
ball_data = [ball.to_numpy() if isinstance(ball, pl.DataFrame) else ball for ball in aligned_balltoball_partitions]

dataset = CricketDataset(team_data, player_data, ball_data, labels)

from sklearn.model_selection import train_test_split

train_indices, temp_indices = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

from torch.utils.data import Subset

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Ensure paths are correctly resolved
data_dir = os.path.join('..', 'data', 'pytorch_data')
os.makedirs(data_dir, exist_ok=True)

# Save Datasets
with open(os.path.join(data_dir, 'train_dataset.pkl'), 'wb') as f:
    pickle.dump(train_dataset, f)

with open(os.path.join(data_dir, 'val_dataset.pkl'), 'wb') as f:
    pickle.dump(val_dataset, f)

with open(os.path.join(data_dir, 'test_dataset.pkl'), 'wb') as f:
    pickle.dump(test_dataset, f)