import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import numpy as np
import polars as pl
from typing import List, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CricketDataset(Dataset):
    def __init__(self, team_stats_list: List[np.ndarray], player_stats_list: List[np.ndarray],
                 ball_stats_list: List[np.ndarray], labels: List[int]) -> None:
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
        return len(self.team_stats_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        team_input = torch.tensor(self.team_stats_list[idx], dtype=torch.float32)
        team_input = team_input.squeeze()
        player_input = torch.tensor(self.player_stats_list[idx], dtype=torch.float32)
        ball_stats = torch.tensor(self.ball_stats_list[idx], dtype=torch.float32)
        ball_input = ball_stats
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return team_input, player_input, ball_input, label


def collate_fn_with_padding(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to pad sequences and stack them into a batch.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]): List of tuples containing team, player, ball data, and labels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Padded and stacked team, player, ball data, and labels.
    """
    team_batch, player_batch, ball_batch, labels = zip(*batch)
    team_batch = [team.clone().detach() if torch.is_tensor(team) else torch.tensor(team) for team in team_batch]
    player_batch = [player.clone().detach() if torch.is_tensor(player) else torch.tensor(player) for player in player_batch]
    ball_batch = [ball.clone().detach() if torch.is_tensor(ball) else torch.tensor(ball) for ball in ball_batch]

    team_batch = pad_sequence(team_batch, batch_first=True, padding_value=0)
    player_batch = pad_sequence(player_batch, batch_first=True, padding_value=0)
    ball_batch = pad_sequence(ball_batch, batch_first=True, padding_value=0)

    labels = torch.tensor(labels).float().unsqueeze(1)
    return team_batch, player_batch, ball_batch, labels

def partition_data_with_keys(df: pl.DataFrame, group_keys: List[str]) -> Tuple[List[Tuple], List[np.ndarray]]:
    """
    Partitions the dataframe by group keys and returns the keys and partitions.

    Args:
        df: Dataframe to be partitioned.
        group_keys: Keys to group by.

    Returns:
        Tuple[List[Tuple], List[np.ndarray]]: List of unique keys and list of partitions.
    """
    partitions = df.partition_by(group_keys)
    keys = [tuple(partition.select(group_keys).unique().to_numpy()[0]) for partition in partitions]
    partitions = [partition.drop(group_keys).to_numpy() for partition in partitions]
    return keys, partitions

def extract_data(dataset) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
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

    for idx in dataset.indices:
        team_input, player_input, ball_input, label = dataset.dataset[idx]
        team_stats_list.append(team_input.numpy())
        player_stats_list.append(player_input.numpy())
        ball_stats_list.append(ball_input.numpy())
        labels_list.append(label.item())
    
    return team_stats_list, player_stats_list, ball_stats_list, labels_list

def augment_match_data(balltoball: List[np.ndarray], team_stats: List[np.ndarray], player_stats: List[np.ndarray],
                       labels: List[int], window_sizes: List[int] = [20, 25, 30, 35, 40, 45]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
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
    augmented_balltoball = []
    augmented_team_stats = []
    augmented_player_stats = []
    augmented_labels = []

    for i in range(len(balltoball)):
        match_length = balltoball[i].shape[0]  # Total number of balls in the match
        for window in window_sizes:
            window_length = window * 6  # Convert overs to balls
            if match_length >= window_length:
                # Slice the data
                augmented_balltoball.append(balltoball[i][:window_length])
                augmented_team_stats.append(team_stats[i])
                augmented_player_stats.append(player_stats[i])
                augmented_labels.append(labels[i])  # Same label for augmented samples
            else:
                augmented_balltoball.append(balltoball[i])
                augmented_team_stats.append(team_stats[i])
                augmented_player_stats.append(player_stats[i])
                augmented_labels.append(labels[i])
    
    return augmented_balltoball, augmented_team_stats, augmented_player_stats, augmented_labels

# Model Architecture

class TeamEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.5) -> None:
        """
        Initializes the TeamEncoder.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden layer.
            dropout (float): Dropout rate.
        """
        super(TeamEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TeamEncoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class PlayerEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_size: int, dropout: float = 0.5) -> None:
        """
        Initializes the PlayerEncoder.

        Args:
            input_channels (int): Number of input channels.
            hidden_size (int): Size of the hidden layer.
            dropout (float): Dropout rate.
        """
        super(PlayerEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PlayerEncoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.to(device)  # Ensure input tensor is on the same device
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        if not hasattr(self, 'fc1'):
            self.fc1 = nn.Linear(x.size(1), self.hidden_size).to(device)  # Ensure fc1 is on the same device
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class BallEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.5) -> None:
        """
        Initializes the BallEncoder.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden layer.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """
        super(BallEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BallEncoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, (hidden, _) = self.rnn(x, (h0, c0))
        hidden = self.dropout(hidden)
        return hidden

class Decoder(nn.Module):   
    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.5) -> None:
        """
        Initializes the Decoder.

        Args:
            input_size (int): Size of the input features.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super(Decoder, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class EncoderDecoderModel(nn.Module):
    def __init__(self, team_input_size: int, player_input_channels: int, ball_input_size: int, hidden_size: int,
                 num_layers: int, num_classes: int, dropout: float = 0.5) -> None:
        """
        Initializes the EncoderDecoderModel.

        Args:
            team_input_size (int): Size of the team input features.
            player_input_channels (int): Number of player input channels.
            ball_input_size (int): Size of the ball input features.
            hidden_size (int): Size of the hidden layer.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super(EncoderDecoderModel, self).__init__()
        self.team_encoder = TeamEncoder(team_input_size, hidden_size, dropout)
        self.player_encoder = PlayerEncoder(player_input_channels, hidden_size, dropout)
        self.ball_encoder = BallEncoder(ball_input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size * 3, num_classes, dropout)

    def forward(self, team: torch.Tensor, player: torch.Tensor, ball: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EncoderDecoderModel.

        Args:
            team (torch.Tensor): Team input tensor.
            player (torch.Tensor): Player input tensor.
            ball (torch.Tensor): Ball input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        team = team.float()
        player = player.float()
        ball = ball.float()

        team_hidden = self.team_encoder(team)

        if player.dim() == 3:
            player = player.unsqueeze(1)  # Add channel dimension
        player_hidden = self.player_encoder(player)

        if ball.dim() == 2:
            ball = ball.unsqueeze(1)
        ball_hidden = self.ball_encoder(ball)[-1]

        combined_hidden = torch.cat((team_hidden, player_hidden, ball_hidden), dim=1)
        output = self.decoder(combined_hidden)
        return output


# Plot

import matplotlib.pyplot as plt

def plot_training_history(epochs_range, train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()