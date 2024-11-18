import os
import sys
import pickle
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.nn import Transformer

sys.path.append(os.path.join(os.getcwd(), '..'))

# Load the dataloaders
train_dataloader = pickle.load(open(os.path.join(os.getcwd(), '..', "data", "pytorch_data", 'train_dataloader.pkl'), 'rb'))
val_dataloader = pickle.load(open(os.path.join(os.getcwd(), '..', "data", "pytorch_data", 'val_dataloader.pkl'), 'rb'))
test_dataloader = pickle.load(open(os.path.join(os.getcwd(), '..', "data", "pytorch_data", 'test_dataloader.pkl'), 'rb'))

# Function to collect and process all matches
def collect_all_matches(dataloaders):
    all_innings1 = []
    all_innings2 = []
    for dataloader in dataloaders:
        for team_input, player_input, ball_input, labels, mask in dataloader:
            # Flatten the ball input data
            ball_input_flat = ball_input.view(-1, ball_input.size(-1)).numpy()
            data = pl.DataFrame(ball_input_flat, schema=["innings", "ball", "runs", "wickets", "curr_score", "curr_wickets", "overs", "run_rate", "required_run_rate", "target"])
            innings1 = data.filter(data['innings'] == 1)
            innings2 = data.filter(data['innings'] == 2)
            all_innings1.append(innings1)
            all_innings2.append(innings2)
    return all_innings1, all_innings2

# Collect all matches data
all_innings1, all_innings2 = collect_all_matches([train_dataloader, val_dataloader, test_dataloader])

# Combine all innings data
combined_innings1 = pl.concat(all_innings1)
combined_innings2 = pl.concat(all_innings2)
print(combined_innings1.shape, combined_innings2.shape)
# Define the number of balls in the last 5 overs (assuming 6 balls per over)
forecast_balls = 30

# # Total number of balls in innings 1 and innings 2
total_balls_innings1 = len(combined_innings1)
total_balls_innings2 = len(combined_innings2)

# # Calculate training and forecasting indices
train_balls_innings1 = total_balls_innings1
train_balls_innings2 = max(1, total_balls_innings2 - forecast_balls)

# Define the Transformer model class
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, nhead=2):
        super(TransformerModel, self).__init__()
        self.transformer = Transformer(d_model=hidden_size, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True)
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, src, trg):
        src = self.fc_in(src)
        trg = self.fc_in(trg)
        output = self.transformer(src, trg)
        output = self.fc_out(output)
        return output

# Update input_size to match the actual input feature size
input_size = 1  # Since we are using "curr_score" as the input feature
hidden_size = 50
output_size = 1

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TransformerModel(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function to create sequences for Transformer model with smaller batch sizes
def create_sequences(data, sequence_length=10, batch_size=32):
    X, y = [], []
    for i in range(0, len(data) - sequence_length, batch_size):
        X_batch = []
        y_batch = []
        for j in range(i, min(i + batch_size, len(data) - sequence_length)):
            X_batch.append(data[j:j + sequence_length])
            y_batch.append(data[j + sequence_length])
        X.append(np.array(X_batch))
        y.append(np.array(y_batch))
    return X, y

# Prepare training data
train_y_innings1 = combined_innings1.to_pandas()["curr_score"].values
train_y_innings2 = combined_innings2.to_pandas()["curr_score"].values[:train_balls_innings2]

scaler_innings1 = MinMaxScaler()
train_y_innings1_scaled = scaler_innings1.fit_transform(train_y_innings1.reshape(-1, 1))

scaler_innings2 = MinMaxScaler()
train_y_innings2_scaled = scaler_innings2.fit_transform(train_y_innings2.reshape(-1, 1))

sequence_length = 10
batch_size = 32  # Adjust batch size to fit in GPU memory
X_innings1_batches, y_innings1_batches = create_sequences(train_y_innings1_scaled.flatten(), sequence_length, batch_size)
X_innings2_batches, y_innings2_batches = create_sequences(train_y_innings2_scaled.flatten(), sequence_length, batch_size)

# Convert to tensors and ensure they are 3-D
X_innings1_batches = [torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device) for X in X_innings1_batches]  # [batch, seq_len, 1]
y_innings1_batches = [torch.tensor(y, dtype=torch.float32).unsqueeze(1).unsqueeze(-1).to(device) for y in y_innings1_batches]  # [batch, 1, 1]
X_innings2_batches = [torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device) for X in X_innings2_batches]  # [batch, seq_len, 1]
y_innings2_batches = [torch.tensor(y, dtype=torch.float32).unsqueeze(1).unsqueeze(-1).to(device) for y in y_innings2_batches]  # [batch, 1, 1]

# Training function with batch processing
def train_model(model, X_batches, y_batches, optimizer, criterion, epochs=100):
    for epoch in range(epochs):
        model.train()
        for X, y in zip(X_batches, y_batches):
            optimizer.zero_grad()
            outputs = model(X, y)  # outputs: [batch, seq_len, 1]
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Train the model
train_model(model, X_innings1_batches, y_innings1_batches, optimizer, criterion, epochs=10)

# In the forecasting function, adjust input data shape
def forecast_remaining_overs(model, innings1_data, innings2_data, steps, sequence_length):
    model.eval()
    input_seq = np.hstack((innings1_data, innings2_data))[-sequence_length:]
    input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # [1, seq_len, 1]
    predictions = []
    with torch.no_grad():
        for _ in range(steps):
            # For forecasting, trg should have shape [batch, 1, 1]
            trg = input_seq[:, -1:, :]  # Take the last time step as trg
            output = model(input_seq, trg)  # trg: [1, 1, 1]
            next_value = output[:, -1, :].squeeze().item()  # Get the last output in the sequence
            predictions.append(next_value)
            # Prepare the next input by appending the predicted value
            next_input = torch.tensor([[[next_value]]], dtype=torch.float32).to(device)  # [1, 1, 1]
            input_seq = torch.cat((input_seq[:, 1:, :], next_input), dim=1)  # Slide the window
    return np.array(predictions)

# Forecast remaining overs
forecast_horizon = total_balls_innings2 - train_balls_innings2
forecast_y_innings2 = forecast_remaining_overs(
    model,
    train_y_innings1_scaled.flatten(),
    train_y_innings2_scaled.flatten(),
    forecast_horizon,
    sequence_length
)

# Inverse scale the forecasted values
forecast_y_innings2 = scaler_innings2.inverse_transform(
    forecast_y_innings2.reshape(-1, 1)
).flatten()

# Generate x-axis for forecast
forecast_x = np.arange(train_balls_innings2 + 1, total_balls_innings2 + 1)

# Plot actual and forecasted values
fig, ax = plt.subplots(figsize=(12, 6))

# Ground truth for innings 1 and innings 2
ax.plot(
    np.arange(1, len(train_y_innings1) + 1), train_y_innings1,
    label="Ground Truth Innings 1", linestyle="-", color="blue"
)
ax.plot(
    np.arange(1, len(train_y_innings2) + 1), train_y_innings2,
    label="Ground Truth Innings 2", linestyle="-", color="orange"
)

# Forecasted values
ax.plot(
    forecast_x, forecast_y_innings2,
    label="Forecast Innings 2 (Seq2Seq)", linestyle="--", color="orange"
)

# Plot settings
ax.set_title("Score Progression and Forecast using Seq2Seq (PyTorch)")
ax.set_xlabel("Ball Number")
ax.set_ylabel("Score")
ax.legend()
ax.grid(True)

plt.show()

# Determine if 2nd innings will win
final_score_innings1 = train_y_innings1[-1] if len(train_y_innings1) > 0 else 0
print(f"Final score of innings 1: {final_score_innings1}")

if forecast_y_innings2[-1] > final_score_innings1:
    print(f"2nd innings will likely win. Predicted Final Score: {forecast_y_innings2[-1]:.2f}")
else:
    print(f"2nd innings will likely not win. Predicted Final Score: {forecast_y_innings2[-1]:.2f}")

# Function to forecast remaining overs for a single match
def forecast_remaining_overs_single(model, innings1_data, innings2_data, steps, sequence_length):
    model.eval()
    input_seq = np.hstack((innings1_data, innings2_data))[-sequence_length:]
    input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # [1, seq_len, 1]
    predictions = []
    with torch.no_grad():
        for _ in range(steps):
            trg = input_seq[:, -1:, :]  # Take the last time step as trg
            output = model(input_seq, trg)  # trg: [1, 1, 1]
            next_value = output[:, -1, :].squeeze().item()  # Get the last output in the sequence
            predictions.append(next_value)
            next_input = torch.tensor([[[next_value]]], dtype=torch.float32).to(device)  # [1, 1, 1]
            input_seq = torch.cat((input_seq[:, 1:, :], next_input), dim=1)  # Slide the window
    return np.array(predictions)

# Function to evaluate the model on all matches
def evaluate_model_on_matches(model, all_innings1, all_innings2, forecast_balls, sequence_length):
    actual_labels = []
    predicted_labels = []
    for innings1, innings2 in zip(all_innings1, all_innings2):
        if len(innings2) < forecast_balls:
            continue  # Skip matches with insufficient data
        train_y_innings1 = innings1.to_pandas()["curr_score"].values
        train_y_innings2 = innings2.to_pandas()["curr_score"].values[:-forecast_balls]
        actual_final_score_innings2 = innings2.to_pandas()["curr_score"].values[-1]
        
        scaler_innings1 = MinMaxScaler()
        train_y_innings1_scaled = scaler_innings1.fit_transform(train_y_innings1.reshape(-1, 1))
        
        scaler_innings2 = MinMaxScaler()
        train_y_innings2_scaled = scaler_innings2.fit_transform(train_y_innings2.reshape(-1, 1))
        
        forecast_horizon = forecast_balls
        forecast_y_innings2 = forecast_remaining_overs_single(
            model,
            train_y_innings1_scaled.flatten(),
            train_y_innings2_scaled.flatten(),
            forecast_horizon,
            sequence_length
        )
        
        forecast_y_innings2 = scaler_innings2.inverse_transform(
            forecast_y_innings2.reshape(-1, 1)
        ).flatten()
        
        final_score_innings1 = train_y_innings1[-1] if len(train_y_innings1) > 0 else 0
        actual_labels.append(int(actual_final_score_innings2 > final_score_innings1))
        predicted_labels.append(int(forecast_y_innings2[-1] > final_score_innings1))
    
    return actual_labels, predicted_labels

# Evaluate the model on all matches
actual_labels, predicted_labels = evaluate_model_on_matches(
    model, all_innings1, all_innings2, forecast_balls, sequence_length
)

# Calculate confusion matrix and accuracy
cm = confusion_matrix(actual_labels, predicted_labels)
accuracy = accuracy_score(actual_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy * 100:.2f}%")
