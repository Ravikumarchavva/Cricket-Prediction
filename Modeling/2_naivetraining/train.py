import os
import sys
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

sys.path.append(os.path.join(os.getcwd(), '..'))
from model_utils import CricketDataset, collate_fn_with_padding
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

# Step 3: Define Model
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
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

# Step 6: Evaluate Model on Test Data
model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
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