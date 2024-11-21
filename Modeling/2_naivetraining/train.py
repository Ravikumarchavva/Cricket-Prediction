import os
import sys
import pickle
import torch
import torch.nn as nn
import wandb
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd

# Initialize Weights & Biases
wandb.init(project="T20I")

sys.path.append(os.path.join(os.getcwd(), '..'))
from model_utils import collate_fn_with_padding
from torch.utils.data import DataLoader

# Load the Datasets
train_dataset = pickle.load(open(os.path.join(os.getcwd(), '..', "data", "pytorch_data", 'train_dataset.pkl'), 'rb'))
val_dataset = pickle.load(open(os.path.join(os.getcwd(), '..', "data", "pytorch_data", 'val_dataset.pkl'), 'rb'))
test_dataset = pickle.load(open(os.path.join(os.getcwd(), '..', "data", "pytorch_data", 'test_dataset.pkl'), 'rb'))

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_with_padding)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_with_padding)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_with_padding)

# Step 1: Extract Data from DataLoader
team_data = []
player_data = []
ball_data = []

for team, player, ball, _ in train_dataloader:
    team_data.append(team.numpy())
    player_data.append(player.numpy())
    ball_data.append(ball.numpy())

# Flatten the lists
team_data = [item for sublist in team_data for item in sublist]
player_data = [item for sublist in player_data for item in sublist]
ball_data = [item for sublist in ball_data for item in sublist]

# Reshape the data to 2D arrays
team_data = [team.reshape(1, -1) if team.ndim == 1 else team for team in team_data]
player_data = [player.reshape(1, -1) if player.ndim == 1 else player for player in player_data]
ball_data = [ball.reshape(1, -1) if ball.ndim == 1 else ball for ball in ball_data]

# Step 2: Normalize Data
scaler_team = StandardScaler()
scaler_player = StandardScaler()
scaler_ball = StandardScaler()

team_data = [scaler_team.fit_transform(team) for team in team_data]
player_data = [scaler_player.fit_transform(player) for player in player_data]
ball_data = [scaler_ball.fit_transform(ball) for ball in ball_data]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 3: Define Model
class TeamEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(TeamEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class PlayerEncoder(nn.Module):
    def __init__(self, input_channels, hidden_size, dropout=0.5):
        super(PlayerEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.to(device)  # Ensure input tensor is on the same device
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        if not hasattr(self, 'fc1'):
            self.fc1 = nn.Linear(x.size(1), self.hidden_size).to(device)  # Ensure fc1 is on the same device
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class BallEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(BallEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, (hidden, _) = self.rnn(x, (h0, c0))
        hidden = self.dropout(hidden)
        return hidden

class Decoder(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.5):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
class EncoderDecoderModel(nn.Module):
    def __init__(self, team_input_size, player_input_channels, ball_input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(EncoderDecoderModel, self).__init__()
        self.team_encoder = TeamEncoder(team_input_size, hidden_size, dropout)
        self.player_encoder = PlayerEncoder(player_input_channels, hidden_size, dropout)
        self.ball_encoder = BallEncoder(ball_input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size * 3, num_classes, dropout)

    def forward(self, team, player, ball):
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

model = EncoderDecoderModel(
    team_input_size=team_data[0].shape[1],
    player_input_channels=1,  # Assuming player data is 2D and needs a channel dimension
    ball_input_size=ball_data[0].shape[1],
    hidden_size=64,
    num_layers=2,
    num_classes=1,
    dropout=0.5
).to(device)  # Move model to device
print(f"Team input size: {team_data[0].shape}")
print(f"ball input size: {ball_data[0].shape}")
print(f"Player input size: {player_data[0].shape}")

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)

# Define the directory to save the best model and plots
save_dir = os.path.dirname(os.path.abspath(__file__))

# Step 4: Train Model
best_val_loss = float('inf')
patience = 10
trigger_times = 0
num_epochs = 100

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
        team, player, ball, labels = team.to(device), player.to(device), ball.to(device), labels.to(device)  # Move data to device
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
            team, player, ball, labels = team.to(device), player.to(device), ball.to(device), labels.to(device)  # Move data to device
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

    # Log metrics to Weights & Biases
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "train_accuracy": train_acc,
        "val_loss": avg_val_loss,
        "val_accuracy": val_acc
    })

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        # Save model checkpoint to Weights & Biases
        wandb.save(os.path.join(save_dir, 'best_model.pth'))
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!')
            break

# Step 5: Plot Training History
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
plt.savefig(os.path.join(save_dir, 'training_history.png'))
plt.show()

# Log final plots to Weights & Biases
wandb.log({"training_history": wandb.Image(os.path.join(save_dir, 'training_history.png'))})

# Step 6: Evaluate Model on Test Data
model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
model.to(device)  # Move model to device
model.eval()
all_labels = []
all_predictions = []
all_probs = []

with torch.no_grad():
    correct = 0
    total = 0
    for team, player, ball, labels in test_dataloader:
        team, player, ball, labels = team.to(device), player.to(device), ball.to(device), labels.to(device)  # Move data to device
        team = team.float()
        player = player.float()
        ball = ball.float()
        labels = labels.float()
        
        outputs = model(team, player, ball)
        probs = outputs.squeeze().cpu().numpy()
        if probs.ndim == 0:
            probs = probs.reshape(1)  # Convert scalar to 1D array
        predicted = (outputs.data > 0.5).float()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probs.extend(probs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Test Accuracy: {:.2f} %'.format(100 * correct / total))

# Step 7: Generate Evaluation Metrics
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
plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
plt.show()

# Convert confusion matrix to DataFrame for logging
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Class 0', 'Actual Class 1'], columns=['Predicted Class 0', 'Predicted Class 1'])

# Log evaluation metrics to Weights & Biases
wandb.log({
    "confusion_matrix": wandb.Table(dataframe=conf_matrix_df),
    "classification_report": class_report,
})

# Finish the Weights & Biases run
wandb.finish()