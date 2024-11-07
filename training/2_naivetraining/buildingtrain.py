import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Team stats model
class TeamStatsModel(nn.Module):
    def __init__(self):
        super(TeamStatsModel, self).__init__()
        self.fc1 = nn.Linear(config['team_model']['input_dim'], config['team_model']['hidden_dim'])
        self.bn1 = nn.BatchNorm1d(config['team_model']['hidden_dim'])
        self.fc2 = nn.Linear(config['team_model']['hidden_dim'], config['team_model']['output_dim'])
        self.bn2 = nn.BatchNorm1d(config['team_model']['output_dim'])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        return x

# Player stats model
class PlayerStatsModel(nn.Module):
    def __init__(self):
        super(PlayerStatsModel, self).__init__()
        self.conv1 = nn.Conv2d(1, config['player_model']['conv1_out_channels'], kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(config['player_model']['conv1_out_channels'])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(config['player_model']['conv1_out_channels'], config['player_model']['conv2_out_channels'], kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(config['player_model']['conv2_out_channels'])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Calculate the size of the flattened layer
        self.flatten_input_dim = self._get_flatten_input_dim(config['player_model']['input_dim'])
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_input_dim, config['player_model']['fc1_output_dim'])
        self.bn3 = nn.BatchNorm1d(config['player_model']['fc1_output_dim'])
        self.dropout = nn.Dropout(0.5)

    def _get_flatten_input_dim(self, input_dim):
        x = torch.zeros(1, *input_dim)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        return x.numel()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        return x

# Ball by ball model
class BallByBallModel(nn.Module):
    def __init__(self):
        super(BallByBallModel, self).__init__()
        self.input_projection = nn.Linear(config['ball_model']['input_dim'], config['ball_model']['transformer']['d_model'])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['ball_model']['transformer']['d_model'],
                nhead=config['ball_model']['transformer']['nhead'],
                dim_feedforward=config['ball_model']['transformer']['dim_feedforward'],
                dropout=config['ball_model']['transformer']['dropout'],
                batch_first=True  # Set batch_first to True
            ),
            num_layers=config['ball_model']['transformer']['num_encoder_layers']
        )
        self.fc1 = nn.Linear(config['ball_model']['transformer']['d_model'], config['ball_model']['output_dim'])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.input_projection(x)
        # No need to permute the dimensions as batch_first=True
        x = self.transformer(x)
        x = x.mean(dim=1)  # Average over sequence_length
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x

# Combined model
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.team_model = TeamStatsModel()
        self.player_model = PlayerStatsModel()
        self.ball_model = BallByBallModel()
        self.fc1 = nn.Linear(config['combined_model']['team_model_output_dim'] + config['combined_model']['player_model_output_dim'] + config['combined_model']['ball_model_output_dim'], config['combined_model']['fc1_output_dim'])
        self.bn1 = nn.BatchNorm1d(config['combined_model']['fc1_output_dim'])
        self.fc2 = nn.Linear(config['combined_model']['fc1_output_dim'], config['combined_model']['fc2_output_dim'])
        self.dropout = nn.Dropout(0.5)

    def forward(self, team, player, ball):
        team = team.squeeze(1)  # Remove the extra dimension
        team = self.team_model(team)
        player = self.player_model(player)
        ball = self.ball_model(ball)
        
        # Reshape team tensor to match dimensions
        team = team.view(team.size(0), -1)
        
        # Flatten player tensor to match dimensions
        player = player.view(player.size(0), -1)
        
        x = torch.cat((team, player, ball), dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

model = CombinedModel().to(device)  # Ensure the model is moved to the GPU
# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Add weight decay for regularization

def calculate_accuracy(output, labels):
    preds = (output > 0.5).float()
    correct = (preds == labels).float().sum()
    return correct / labels.numel()

epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_accuracy = 0
    for i, (team, player, ball, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
        team, player, ball, labels = team.to(device), player.to(device), ball.to(device), labels.to(device)
        
        # Reshape labels to match the model's output shape
        labels = labels.view(-1, 1)
        
        optimizer.zero_grad()
        output = model(team, player, ball)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_accuracy += calculate_accuracy(output, labels).item()
    print(f'Epoch: {epoch+1}, Training Loss: {train_loss/len(train_loader)}, Training Accuracy: {train_accuracy/len(train_loader)}')
    
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for i, (team, player, ball, labels) in enumerate(val_loader):
            team, player, ball, labels = team.to(device), player.to(device), ball.to(device), labels.to(device)
            
            # Reshape labels to match the model's output shape
            labels = labels.view(-1, 1)
            
            output = model(team, player, ball)
            loss = criterion(output, labels)
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(output, labels).item()
    print(f'Epoch: {epoch+1}, Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy/len(val_loader)}')