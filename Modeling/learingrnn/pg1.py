import os
import sys
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

sys.path.append(os.path.join(os.getcwd(), ".."))
from model_utils import CricketDataset, collate_fn_with_padding

# Step 1: Load Data
def load_data():
    balltoball = pl.read_csv(os.path.join(os.path.join('..', "data", "filtered_data", "balltoball.csv")))
    team_stats = pl.read_csv(os.path.join(os.path.join('..', "data", "filtered_data", "team12_stats.csv")))
    players_stats = pl.read_csv(os.path.join(os.path.join('..', "data", "filtered_data", "players_stats.csv")))
    return balltoball, team_stats, players_stats

balltoball, team_stats, players_stats = load_data()

# Step 2: Partition Data
def partition_data_with_keys(df, group_keys):
    partitions = df.partition_by(group_keys)
    keys = [tuple(partition.select(group_keys).unique().to_numpy()[0]) for partition in partitions]
    partitions = [partition.drop(group_keys).to_numpy() for partition in partitions]
    return keys, partitions

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

train_indices, temp_indices = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

print(f'Number of samples: {len(dataset)}')
print(f'Number of training samples: {len(train_dataset)}')
print(f'Number of validation samples: {len(val_dataset)}')
print(f'Number of test samples: {len(test_dataset)}')

# Step 5: Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_with_padding)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_with_padding)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_with_padding)

# Step 6: Normalize Data
scaler_team = StandardScaler()
scaler_player = StandardScaler()
scaler_ball = StandardScaler()

team_data = [scaler_team.fit_transform(team) for team in team_data]
player_data = [scaler_player.fit_transform(player) for player in player_data]
ball_data = [scaler_ball.fit_transform(ball) for ball in ball_data]

# Step 7: Define Model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, (hidden, _) = self.rnn(x, (h0, c0))
        return hidden

class Decoder(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class EncoderDecoderModel(nn.Module):
    def __init__(self, team_input_size, player_input_size, ball_input_size, hidden_size, num_layers, num_classes):
        super(EncoderDecoderModel, self).__init__()
        self.team_encoder = Encoder(team_input_size, hidden_size, num_layers)
        self.player_encoder = Encoder(player_input_size, hidden_size, num_layers)
        self.ball_encoder = Encoder(ball_input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size * 3, num_classes)

    def forward(self, team, player, ball):
        team = team.float()
        player = player.float()
        ball = ball.float()

        if team.dim() == 2:
            team = team.unsqueeze(1)
        team_hidden = self.team_encoder(team)[-1]

        if player.dim() == 2:
            player = player.unsqueeze(1)
        player_hidden = self.player_encoder(player)[-1]

        if ball.dim() == 2:
            ball = ball.unsqueeze(1)
        ball_hidden = self.ball_encoder(ball)[-1]

        combined_hidden = torch.cat((team_hidden, player_hidden, ball_hidden), dim=1)
        output = self.decoder(combined_hidden)
        return output

model = EncoderDecoderModel(
    team_input_size=team_data[0].shape[1],
    player_input_size=player_data[0].shape[1],
    ball_input_size=ball_data[0].shape[1],
    hidden_size=64,
    num_layers=2,
    num_classes=1
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Step 8: Train Model
best_val_loss = float('inf')
patience = 10
trigger_times = 0
num_epochs = 25

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for team, player, ball, labels in tqdm(train_dataloader):
        labels = labels.float()
        outputs = model(team, player, ball)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = (outputs.data > 0.5).float()
        total += labels.size(0)
        running_corrects += (predicted == labels).sum().item()
    avg_loss = running_loss / len(train_dataloader)
    train_acc = 100 * running_corrects / total

    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)

    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_total = 0
    with torch.no_grad():
        for team, player, ball, labels in val_dataloader:
            labels = labels.float()
            outputs = model(team, player, ball)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = (outputs.data > 0.5).float()
            val_total += labels.size(0)
            val_corrects += (predicted == labels).sum().item()
    avg_val_loss = val_loss / len(val_dataloader)
    val_acc = 100 * val_corrects / val_total

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!')
            break

# Step 9: Plot Training History
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy History')
plt.legend()

plt.tight_layout()
plt.show()

# Step 10: Evaluate Model on Test Data
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
all_labels = []
all_predictions = []
all_probs = []

with torch.no_grad():
    correct = 0
    total = 0
    for team, player, ball, labels in test_dataloader:
        team = team.float()
        player = player.float()
        ball = ball.float()
        labels = labels.float()
        
        outputs = model(team, player, ball)
        probs = outputs.squeeze().cpu().numpy()
        predicted = (outputs.data > 0.5).float()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probs.extend(probs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Test Accuracy: {:.2f} %'.format(100 * correct / total))

# Step 11: Generate Evaluation Metrics
conf_matrix = confusion_matrix(all_labels, all_predictions)
print('Confusion Matrix:')
print(conf_matrix)

class_report = classification_report(all_labels, all_predictions, target_names=['Class 0', 'Class 1'])
print('Classification Report:')
print(class_report)

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()