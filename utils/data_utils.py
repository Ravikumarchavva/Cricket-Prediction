import torch
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import wandb
import os
import polars as pl

# Data


class CricketDataset(Dataset):
    def __init__(
        self,
        team_stats_list: List[np.ndarray],
        player_stats_list: List[np.ndarray],
        ball_stats_list: List[np.ndarray],
        labels: List[int],
    ) -> None:
        """
        Initializes the CricketDataset.

        Args:
            team_stats_list (List[np.ndarray]): List of team statistics arrays.
            player_stats_list (List[np.ndarray]): List of player statistics arrays.
            ball_stats_list (List[np.ndarray]): List of ball-by-ball data arrays.
            labels (List[int]): List of match outcome labels.
        """
        self.team_stats_list = team_stats_list
        self.player_stats_list = player_stats_list
        self.ball_stats_list = ball_stats_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        team_input = torch.tensor(self.team_stats_list[idx], dtype=torch.float32)
        team_input = team_input.squeeze()
        player_input = torch.tensor(self.player_stats_list[idx], dtype=torch.float32)
        ball_stats = torch.tensor(self.ball_stats_list[idx], dtype=torch.float32)
        ball_input = ball_stats
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return team_input, player_input, ball_input, label


def collate_fn_with_padding(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to pad sequences and stack them into a batch.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]): List of tuples containing team, player, ball data, and labels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Padded and stacked team, player, ball data, and labels.
    """
    team_batch, player_batch, ball_batch, labels = zip(*batch)
    team_batch = [
        team.clone().detach() if torch.is_tensor(team) else torch.tensor(team)
        for team in team_batch
    ]
    player_batch = [
        player.clone().detach() if torch.is_tensor(player) else torch.tensor(player)
        for player in player_batch
    ]
    ball_batch = [
        ball.clone().detach() if torch.is_tensor(ball) else torch.tensor(ball)
        for ball in ball_batch
    ]

    team_batch = pad_sequence(team_batch, batch_first=True, padding_value=0)
    player_batch = pad_sequence(player_batch, batch_first=True, padding_value=0)
    ball_batch = pad_sequence(ball_batch, batch_first=True, padding_value=0)

    labels = torch.tensor(labels).float().unsqueeze(1)
    return team_batch, player_batch, ball_batch, labels


def partition_data_with_keys(
    df: pl.DataFrame, group_keys: List[str]
) -> Tuple[List[Tuple], List[np.ndarray]]:
    """
    Partitions the dataframe by group keys and returns the keys and partitions.

    Args:
        df: Dataframe to be partitioned.
        group_keys: Keys to group by.

    Returns:
        Tuple[List[Tuple], List[np.ndarray]]: List of unique keys and list of partitions.
    """
    partitions = df.partition_by(group_keys)
    keys = [
        tuple(partition.select(group_keys).unique().to_numpy()[0])
        for partition in partitions
    ]
    partitions = [partition.drop(group_keys).to_numpy() for partition in partitions]
    return keys, partitions


def dataset_to_list(
    dataset: Dataset,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Extracts data from the dataset.

    Args:
        dataset: Dataset to extract data from.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]: Extracted team, player, ball data, and labels.
    """
    team_stats_list = []
    player_stats_list = []
    ball_stats_list = []
    labels_list = []

    for data in dataset:
        team_input, player_input, ball_input, label = data
        team_stats_list.append(team_input.numpy())
        player_stats_list.append(player_input.numpy())
        ball_stats_list.append(ball_input.numpy())
        labels_list.append(label.item())

    return team_stats_list, player_stats_list, ball_stats_list, labels_list


def augument_function(
    balltoball: List[np.ndarray],
    team_stats: List[np.ndarray],
    player_stats: List[np.ndarray],
    labels: List[int],
    window_sizes: List[int] = np.arange(20,46,5),
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Augments data to simulate predictions at various match stages using sliding windows.

    Args:
        balltoball (List[np.ndarray]): List of ball-by-ball data arrays.
        team_stats (List[np.ndarray]): List of team statistics arrays.
        player_stats (List[np.ndarray]): List of player statistics arrays.
        labels (List[int]): List of match outcome labels.
        window_sizes (List[int]): List of window sizes (in overs).

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]: Augmented balltoball, team_stats, player_stats, and labels.
    """
    augmented_team_stats = []
    augmented_player_stats = []
    augmented_balltoball = []
    augmented_labels = []

    for i in range(len(balltoball)):
        match_length = balltoball[i].shape[0]  # Total number of balls in the match
        for window in window_sizes:
            window_length = window * 6  # Convert overs to balls
            if match_length >= window_length:
                # Slice the data
                augmented_team_stats.append(team_stats[i])
                augmented_player_stats.append(player_stats[i])
                augmented_balltoball.append(balltoball[i][:window_length])
                augmented_labels.append(labels[i])  # Same label for augmented samples
            else:
                augmented_team_stats.append(team_stats[i])
                augmented_player_stats.append(player_stats[i])
                augmented_balltoball.append(balltoball[i])
                augmented_labels.append(labels[i])

    return (
        augmented_team_stats,
        augmented_player_stats,
        augmented_balltoball,
        augmented_labels,
    )


def load_dataset(file_path: str) -> CricketDataset:
    # Load the saved data
    saved_data = torch.load(file_path, weights_only=False)

    # Recreate the CricketDataset
    return CricketDataset(
        team_stats_list=saved_data["team_stats"],
        player_stats_list=saved_data["player_stats"],
        ball_stats_list=saved_data["ball_stats"],
        labels=saved_data["labels"],
    )


def load_datasets(
    entity_name: str = "ravikumarchavva-org",
    project: str = "T20I-CRICKET-WINNER-PREDICTION",
    dataset_name: str = "cricket-dataset",
    version: str = "latest",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load datasets from W&B artifacts.

    Args:
        entity_name (str): W&B entity name.
        project (str): W&B project name.
        dataset_name (str): Name of the dataset artifact.
        version (str): Version of the dataset artifact to load.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Training, validation, and test datasets.
    """
    # Ensure wandb is initialized
    if not wandb.run:
        wandb.init(project=project, entity=entity_name)
    # Use the artifact
    artifact = wandb.run.use_artifact(
        f"{entity_name}/{project}/{dataset_name}:{version}"
    )

    # Load datasets directly into memory and remove files after loading
    train_path = artifact.get_entry("train_dataset.pt").download()
    train_dataset = load_dataset(train_path)
    os.remove(train_path)  # Clean up

    val_path = artifact.get_entry("val_dataset.pt").download()
    val_dataset = load_dataset(val_path)
    os.remove(val_path)  # Clean up

    test_path = artifact.get_entry("test_dataset.pt").download()
    test_dataset = load_dataset(test_path)
    os.remove(test_path)  # Clean up

    return train_dataset, val_dataset, test_dataset


def augument_data(
    train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Augment the data by creating multiple samples for each match at different stages

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Augmented training, validation, and test datasets
    """

    train_team_data, train_player_data, train_ball_data, train_labels = dataset_to_list(
        train_dataset
    )
    val_team_data, val_player_data, val_ball_data, val_labels = dataset_to_list(
        val_dataset
    )
    test_team_data, test_player_data, test_ball_data, test_labels = dataset_to_list(
        test_dataset
    )

    train_team_data, train_player_data, train_ball_data, train_labels = augument_function(
        train_ball_data, train_team_data, train_player_data, train_labels
    )
    val_team_data, val_player_data, val_ball_data, val_labels = augument_function(
        val_ball_data, val_team_data, val_player_data, val_labels
    )
    test_team_data, test_player_data, test_ball_data, test_labels = augument_function(
        test_ball_data, test_team_data, test_player_data, test_labels
    )

    train_dataset = CricketDataset(
        train_team_data, train_player_data, train_ball_data, train_labels
    )
    val_dataset = CricketDataset(
        val_team_data, val_player_data, val_ball_data, val_labels
    )
    test_dataset = CricketDataset(
        test_team_data, test_player_data, test_ball_data, test_labels
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


def create_datasets(
    train_data: List[np.ndarray],
    val_data: List[np.ndarray],
    test_data: List[np.ndarray],
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Creates CricketDataset instances for training, validation, and testing.

    Args:
        train_data (List[np.ndarray]): Tuple containing augmented training data.
        val_data (List[np.ndarray]): Tuple containing augmented validation data.
        test_data (List[np.ndarray]): Tuple containing augmented test data.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Training, validation, and test datasets.
    """
    # Ensure the correct order: team, player, ball, label
    train_dataset = CricketDataset(*train_data)
    val_dataset = CricketDataset(*val_data)
    test_dataset = CricketDataset(*test_data)
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_padding,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_padding,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_padding,
    )
    return train_dataloader, val_dataloader, test_dataloader
