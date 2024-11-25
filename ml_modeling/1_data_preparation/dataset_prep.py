import os
import sys
import numpy as np
import polars as pl
import torch
from typing import Tuple

sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from utils.data_utils import partition_data_with_keys, CricketDataset


# Step 1: Load Data
def load_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Loads the ball-by-ball data, team statistics, and player statistics.

    Returns:
        Tuple containing ball-by-ball DataFrame, team statistics DataFrame, and player statistics DataFrame.
    """
    balltoball = pl.read_csv(
        os.path.join(os.path.join("..", "filtered_data", "balltoball.csv"))
    )
    team_stats = pl.read_csv(
        os.path.join(os.path.join("..", "filtered_data", "team12_stats.csv"))
    )
    players_stats = pl.read_csv(
        os.path.join(os.path.join("..", "filtered_data", "players_stats.csv"))
    )
    return balltoball, team_stats, players_stats


balltoball, team_stats, players_stats = load_data()

# Step 2: Partition Data

balltoball_keys, balltoball_partitions = partition_data_with_keys(
    balltoball, ["match_id"]
)
team_stats_keys, team_stats_partitions = partition_data_with_keys(
    team_stats, ["match_id"]
)
players_stats_keys, players_stats_partitions = partition_data_with_keys(
    players_stats, ["match_id"]
)


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
    aligned_balltoball_partitions.append(
        balltoball_partition[:-1, :-1]
    )  # remove the last row and column
    aligned_team_stats_partitions.append(team_stats_partition)
    aligned_players_stats_partitions.append(players_stats_partition)
    labels.append(label)

labels = np.array(labels)

# Step 4: Prepare Data for Training
team_data = [
    team.to_numpy() if isinstance(team, pl.DataFrame) else team
    for team in aligned_team_stats_partitions
]
player_data = [
    players.to_numpy() if isinstance(players, pl.DataFrame) else players
    for players in aligned_players_stats_partitions
]
ball_data = [
    ball.to_numpy() if isinstance(ball, pl.DataFrame) else ball
    for ball in aligned_balltoball_partitions
]

dataset = CricketDataset(team_data, player_data, ball_data, labels)

from sklearn.model_selection import train_test_split

train_indices, temp_indices = train_test_split(
    np.arange(len(labels)), test_size=0.2, random_state=42
)
val_indices, test_indices = train_test_split(
    temp_indices, test_size=0.5, random_state=42
)

from torch.utils.data import Subset

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Ensure paths are correctly resolved
data_dir = os.path.join("..", "pytorch_data")
os.makedirs(data_dir, exist_ok=True)

train_path = os.path.join(data_dir, "train_dataset.pt")
val_path = os.path.join(data_dir, "val_dataset.pt")
test_path = os.path.join(data_dir, "test_dataset.pt")


# Prepare data to save subsets
def save_dataset(dataset: Subset, file_path: str):
    # Extract subset data
    team_stats = [dataset.dataset.team_stats_list[i] for i in dataset.indices]
    player_stats = [dataset.dataset.player_stats_list[i] for i in dataset.indices]
    ball_stats = [dataset.dataset.ball_stats_list[i] for i in dataset.indices]
    labels = [dataset.dataset.labels[i] for i in dataset.indices]

    # Save as a dictionary
    torch.save(
        {
            "team_stats": team_stats,
            "player_stats": player_stats,
            "ball_stats": ball_stats,
            "labels": labels,
        },
        file_path,
    )


save_dataset(train_dataset, train_path)
save_dataset(val_dataset, val_path)
save_dataset(test_dataset, test_path)

import wandb

# Step 5: Save datasets to W&B using artifacts
wandb.init(project="T20I-CRICKET-WINNER-PREDICTION", name="dataset-upload")

# Create artifact
artifact = wandb.Artifact(
    "cricket-dataset",
    type="dataset",
    description="Train, validation, and test sets for T20I cricket winner prediction",
    metadata={
        "total_samples": len(labels),
        "train_split": len(train_indices),
        "val_split": len(val_indices),
        "test_split": len(test_indices),
    },
)

# Add files to artifact
artifact.add_file(train_path)
artifact.add_file(val_path)
artifact.add_file(test_path)

# Log artifact
wandb.log_artifact(artifact)

# Finish W&B run
wandb.finish()
