import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        self,
        input_channels: int,
        hidden_size: int,
        input_height: int,
        input_width: int,
        dropout: float = 0.5,
    ):
        """
        Initializes the PlayerEncoder.

        Args:
            input_channels (int): Number of input channels.
            hidden_size (int): Size of the hidden layer.
            input_height (int): Height of the input.
            input_width (int): Width of the input.
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

        # Compute flattened size dynamically
        self.flattened_size = self._calculate_flattened_size(input_height, input_width)
        self.fc1 = nn.Linear(self.flattened_size, hidden_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _calculate_flattened_size(self, height: int, width: int) -> int:
        """
        Calculates the flattened size after convolutions and pooling.

        Args:
            height (int): Input height.
            width (int): Input width.

        Returns:
            int: Flattened size.
        """
        # After first conv + pool
        height = (height + 2 * 1 - 3) // 1 + 1  # Conv padding=1, kernel_size=3
        height = height // 2  # MaxPool kernel_size=2, stride=2

        width = (width + 2 * 1 - 3) // 1 + 1  # Conv padding=1, kernel_size=3
        width = width // 2  # MaxPool kernel_size=2, stride=2

        # After second conv + pool
        height = (height + 2 * 1 - 3) // 1 + 1
        height = height // 2

        width = (width + 2 * 1 - 3) // 1 + 1
        width = width // 2

        return height * width * 64  # 64 channels from the second convolution layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PlayerEncoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Add input validation
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (batch_size, channels, height, width), got {x.dim()}D"
            )

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.pool(x)

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
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

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
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

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
        player_input_height: int,
        player_input_width: int,
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
            player_input_height (int): Height of the player input.
            player_input_width (int): Width of the player input.
            ball_input_size (int): Size of the ball input features.
            hidden_size (int): Size of the hidden layer.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super(EncoderDecoderModel, self).__init__()
        self.team_encoder = TeamEncoder(team_input_size, hidden_size, dropout)
        self.player_encoder = PlayerEncoder(
            player_input_channels,
            hidden_size,
            player_input_height,
            player_input_width,
            dropout,
        )
        self.ball_encoder = BallEncoder(
            ball_input_size, hidden_size, num_layers, dropout
        )
        self.decoder = Decoder(hidden_size * 3, num_classes, dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        self.team_encoder._initialize_weights()
        self.player_encoder._initialize_weights()
        self.ball_encoder._initialize_weights()
        self.decoder._initialize_weights()

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
            player = player.unsqueeze(1)  # Add channel dimension if missing
        player_hidden = self.player_encoder(player)

        if ball.dim() == 2:
            ball = ball.unsqueeze(1)
        ball_hidden = self.ball_encoder(ball)[-1]

        combined_hidden = torch.cat((team_hidden, player_hidden, ball_hidden), dim=1)
        output = self.decoder(combined_hidden)
        return output

    def to(self, device):
        # Ensure all sub-modules are on the same device
        super().to(device)
        self.team_encoder.to(device)
        self.player_encoder.to(device)
        self.ball_encoder.to(device)
        self.decoder.to(device)
        return self
