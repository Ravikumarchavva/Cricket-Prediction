import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

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

import polars as pl
data = pl.DataFrame(ball_input[0].numpy(),schema=["innings","ball","runs","wickets","overs","run_rate","curr_score","curr_wickets","target"])

innings1 = data.filter(data['innings'] == 1)
innings2 = data.filter(data['innings'] == 2)
print(innings1)
print(innings2)

# Preprocess the data for a single match
def preprocess_single_match(dataloader):
    for team_input, player_input, ball_input, label, mask in dataloader:
        ball_input = ball_input[mask == 1]  # Remove padding
        return ball_input.unsqueeze(0), label.unsqueeze(0)

train_features, train_labels = preprocess_single_match(train_dataloader)
val_features, val_labels = preprocess_single_match(val_dataloader)
test_features, test_labels = preprocess_single_match(test_dataloader)

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_dim = train_features.shape[2]
hidden_dim = 12
output_dim = 1
num_layers = 2

model = RNNModel(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
val_outputs = model(val_features)
val_loss = criterion(val_outputs, val_labels)
print(f'Validation Loss: {val_loss.item():.4f}')

# Predict the score for the 2nd innings
test_outputs = model(test_features)
print(f'Test Predictions: {test_outputs}')