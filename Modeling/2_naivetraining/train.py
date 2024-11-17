import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import pickle


# Load the dataloaders
train_dataloader = pickle.load(open(os.path.join(os.getcwd(), '..', "data", "pytorch_data", 'train_dataloader.pkl'), 'rb'))
val_dataloader = pickle.load(open(os.path.join(os.getcwd(), '..', "data", "pytorch_data", 'val_dataloader.pkl'), 'rb'))
test_dataloader = pickle.load(open(os.path.join(os.getcwd(), '..', "data", "pytorch_data", 'test_dataloader.pkl'), 'rb'))

for team_input, player_input, ball_input, labels, mask in train_dataloader:
    print(f"Team input shape: {team_input.shape}")  # [batch_size, team_feature_dim]
    print(f"Player input shape: {player_input.shape}")  # [batch_size, player_feature_dim]
    print(f"Padded ball input shape: {ball_input.shape}")  # [batch_size, max_seq_len, ball_feature_dim]
    print(f"Mask shape: {mask.shape}")  # [batch_size, max_seq_len]
    print(f"Labels shape: {labels.shape}")  # [batch_size]
    break

# Before training, compute and print class balance
from collections import Counter

# Get all labels from the training data
all_labels = []
for _, _, _, labels, _ in train_dataloader:
    all_labels.extend(labels.numpy())

class_counts = Counter(all_labels)
total_samples = len(all_labels)
print("Class distribution in training data:")
for cls, count in class_counts.items():
    print(f"Class {int(cls)}: {count} samples ({(count/total_samples)*100:.2f}%)")

# Architecture of the model

import torch
import torch.nn as nn
import torch.nn.functional as F
# from tqdm import tqdm  # Removed tqdm import
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Implement Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        # Ensure in_features matches concatenated hidden states (hidden_dim*4)
        self.attn = nn.Linear(self.hidden_dim * 4, self.hidden_dim)  # Changed from hidden_dim * 2 to hidden_dim * 4
        self.v = nn.Parameter(torch.rand(self.hidden_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_dim*2]
        # encoder_outputs: [batch_size, seq_len, hidden_dim*2]
        attn_energies = self.score(hidden, encoder_outputs)  # [batch_size, seq_len]
        return F.softmax(attn_energies, dim=1)  # [batch_size, seq_len]

    def score(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_dim*2]
        # encoder_outputs: [batch_size, seq_len, hidden_dim*2]
        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)  # [batch_size, seq_len, hidden_dim*2]
        # Concatenate hidden and encoder_outputs along the feature dimension
        energy_input = torch.cat((hidden, encoder_outputs), dim=2)  # [batch_size, seq_len, hidden_dim*4]
        energy = torch.tanh(self.attn(energy_input))  # [batch_size, seq_len, hidden_dim]
        energy = energy.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        energy = torch.bmm(v, energy)  # [batch_size, 1, seq_len]
        return energy.squeeze(1)  # [batch_size, seq_len]

# Add TeamStatsModel
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

# Modify PlayerStatsModel to use vertically-oriented convolutional filters
class PlayerStatsModel(nn.Module):
    def __init__(self, input_size, seq_len):
        super(PlayerStatsModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 1))  # Vertical kernel
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1))  # Vertical kernel
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * ((seq_len - 4) // 4), 16)  # Adjust input size dynamically

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, seq_len, input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x

# Update the model to include attention
class CricketModel(nn.Module):
    def __init__(self, team_input_dim, player_input_dim, ball_input_dim, hidden_dim=16, player_seq_len=10):
        super(CricketModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = Attention(hidden_dim)
        
        # Initialize TeamStatsModel
        self.team_stats_model = TeamStatsModel(input_size=team_input_dim)
        
        # Initialize PlayerStatsModel with vertical convolution
        self.player_stats_model = PlayerStatsModel(input_size=player_input_dim, seq_len=player_seq_len)
        
        # Player team aggregation
        self.player_team_fc = nn.Sequential(
            nn.Linear(16 * 11, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Team stats encoding
        self.team_fc = nn.Sequential(
            nn.Linear(16, 32),  # Output from TeamStatsModel is 16
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Ball-by-ball sequence encoding with GRU
        self.rnn = nn.GRU(ball_input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(32 + 32 + hidden_dim * 2, 32),  # hidden_dim * 2 = 32
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, team_input, player_input, ball_input, mask):
        # Team stats encoding
        team_out = self.team_stats_model(team_input)  # Output: [batch_size, 16]
        team_out = self.team_fc(team_out)  # Output: [batch_size, 32]

        # Player stats encoding per team
        team1_player_input = player_input[:, :11, :]  # [batch_size, 11, input_size]
        team2_player_input = player_input[:, 11:, :]  # [batch_size, 11, input_size]

        # Process each team's players
        team1_player_out = self.player_stats_model(team1_player_input)  # [batch_size, 16]
        team2_player_out = self.player_stats_model(team2_player_input)  # [batch_size, 16]

        # Aggregate player features per team
        team1_player_out = self.player_team_fc(team1_player_out)  # [batch_size, 32]
        team2_player_out = self.player_team_fc(team2_player_out)  # [batch_size, 32]

        # Compute difference between team representations
        player_diff = team1_player_out - team2_player_out  # [batch_size, 32]

        # Ball-by-ball sequence encoding with attention
        lengths = mask.sum(1).cpu()
        packed_ball_input = pack_padded_sequence(ball_input, lengths, batch_first=True, enforce_sorted=False)
        packed_rnn_out, hidden = self.rnn(packed_ball_input)
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)  # [batch_size, seq_len, hidden_dim*2]
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # [batch_size, hidden_dim*2]

        # Apply attention
        attn_weights = self.attention(hidden, rnn_out)  # [batch_size, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), rnn_out).squeeze(1)  # [batch_size, hidden_dim*2]

        # Combine all features
        combined_out = torch.cat((team_out, player_diff, context), dim=1)  # [batch_size, 32 + 32 + hidden_dim*2]

        # Classification
        return self.fc(combined_out)

# Initialize the model with adjusted parameters
model = CricketModel(
    team_input_dim=team_input.shape[1],
    player_input_dim=player_input.shape[2],
    ball_input_dim=ball_input.shape[2],
    hidden_dim=16  # Ensure hidden_dim matches the updated value
)

# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Update the loss function
criterion = nn.BCEWithLogitsLoss()

# 2. Adjust the optimizer to include weight decay for regularization
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Added weight_decay

# 3. Add a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Lists to store metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Import additional metrics
from sklearn.metrics import f1_score, precision_score, recall_score
# Import ROC AUC metric
from sklearn.metrics import roc_auc_score

# Training loop
# from tqdm import tqdm  # Removed tqdm import

num_epochs = 50  # Increased number of epochs from 20 to 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    y_true_train = []
    y_pred_train = []
    y_pred_prob_train = []  # Add this list
    # Removed tqdm from the loop
    for team_input, player_input, ball_input, labels, mask in train_dataloader:
        team_input, player_input, ball_input, labels, mask = team_input.to(device), player_input.to(device), ball_input.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad()
        outputs = model(team_input, player_input, ball_input, mask).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        probs = torch.sigmoid(outputs).cpu().detach().numpy()
        predictions = (probs > 0.5).astype(int)
        correct_preds += (predictions == labels.cpu().detach().numpy()).sum()
        total_preds += labels.size(0)

        y_true_train.extend(labels.detach().cpu().detach().numpy())
        y_pred_train.extend(predictions)
        y_pred_prob_train.extend(probs)  # Collect probabilities
    train_loss = running_loss / len(train_dataloader)
    train_acc = correct_preds / total_preds

    # Calculate additional metrics
    train_f1 = f1_score(y_true_train, y_pred_train)
    train_precision = precision_score(y_true_train, y_pred_train, zero_division=0)
    train_recall = recall_score(y_true_train, y_pred_train, zero_division=0)
    train_auc = roc_auc_score(y_true_train, y_pred_prob_train)  # Compute AUC
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation loop with accuracy calculation
    model.eval()
    val_running_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0
    y_true_val = []
    y_pred_val = []
    y_pred_prob_val = []  # Add this list
    with torch.no_grad():
        for team_input, player_input, ball_input, labels, mask in val_dataloader:
            team_input, player_input, ball_input, labels, mask = team_input.to(device), player_input.to(device), ball_input.to(device), labels.to(device), mask.to(device)
            outputs = model(team_input, player_input, ball_input, mask).squeeze()
            loss = criterion(outputs, labels.float())
            val_running_loss += loss.item()
            probs = torch.sigmoid(outputs).cpu().detach().numpy()
            predictions = (probs > 0.5).astype(int)
            val_correct_preds += (predictions == labels.detach().cpu().detach().numpy()).sum()
            val_total_preds += labels.size(0)

            y_true_val.extend(labels.cpu().detach().numpy())
            y_pred_val.extend(predictions)
            y_pred_prob_val.extend(probs)  # Collect probabilities
    val_loss = val_running_loss / len(val_dataloader)
    val_acc = val_correct_preds / val_total_preds

    # Calculate additional metrics
    val_f1 = f1_score(y_true_val, y_pred_val, zero_division=0)
    val_precision = precision_score(y_true_val, y_pred_val, zero_division=0)
    val_recall = recall_score(y_true_val, y_pred_val, zero_division=0)
    val_auc = roc_auc_score(y_true_val, y_pred_prob_val)  # Compute AUC
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, "
          f"Train F1: {train_f1:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")

    # Step the scheduler
    scheduler.step()