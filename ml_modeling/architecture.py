import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Tuple
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return len(self.team_stats_list)

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
    def __init__(
        self, input_channels: int, hidden_size: int, dropout: float = 0.5
    ) -> None:
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
        self.fc1 = nn.Linear(64 * 3 * 5, hidden_size).to(device)  # Adjust input size to 960

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
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


class BallEncoder(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.5
    ) -> None:
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
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
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
    def __init__(
        self,
        team_input_size: int,
        player_input_channels: int,
        ball_input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
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
        self.ball_encoder = BallEncoder(
            ball_input_size, hidden_size, num_layers, dropout
        )
        self.decoder = Decoder(hidden_size * 3, num_classes, dropout)

    def forward(
        self, team: torch.Tensor, player: torch.Tensor, ball: torch.Tensor
    ) -> torch.Tensor:
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
