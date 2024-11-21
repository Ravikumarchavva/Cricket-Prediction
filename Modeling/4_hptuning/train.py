import os
import sys
import pickle
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
import pandas as pd

# Initialize Weights & Biases
wandb.init(project="T20I")

sys.path.append(os.path.join(os.getcwd(), '..'))
from model_utils import collate_fn_with_padding, extract_data, augment_match_data, CricketDataset, EncoderDecoderModel, plot_training_history, plot_roc_curve
from torch.utils.data import DataLoader

# Load the Datasets
train_dataset = pickle.load(open(os.path.join(os.getcwd(), '..', "data", "pytorch_data", 'train_dataset.pkl'), 'rb'))
val_dataset = pickle.load(open(os.path.join(os.getcwd(), '..', "data", "pytorch_data", 'val_dataset.pkl'), 'rb'))
test_dataset = pickle.load(open(os.path.join(os.getcwd(), '..', "data", "pytorch_data", 'test_dataset.pkl'), 'rb'))

# Step 1: Extract Data from Dataset
train_team_data, train_player_data, train_ball_data, train_labels = extract_data(train_dataset)
val_team_data, val_player_data, val_ball_data, val_labels = extract_data(val_dataset)
test_team_data, test_player_data, test_ball_data, test_labels = extract_data(test_dataset)

# Step 2: Augment Data
train_ball_data, train_team_data, train_player_data, train_labels = augment_match_data(train_ball_data, train_team_data, train_player_data, train_labels)
val_ball_data, val_team_data, val_player_data, val_labels = augment_match_data(val_ball_data, val_team_data, val_player_data, val_labels)
test_ball_data, test_team_data, test_player_data, test_labels = augment_match_data(test_ball_data, test_team_data, test_player_data, test_labels)

# Step 3: Convert augmented data to PyTorch Dataset
train_dataset = CricketDataset(train_team_data, train_player_data, train_ball_data, train_labels)
val_dataset = CricketDataset(val_team_data, val_player_data, val_ball_data, val_labels)
test_dataset = CricketDataset(test_team_data, test_player_data, test_ball_data, test_labels) 

# Step 4: Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_with_padding)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_with_padding)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_with_padding)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 3: Define Model

model = EncoderDecoderModel(
    team_input_size=train_team_data[0].shape[0],
    player_input_channels=1,  # Assuming player data is 2D and needs a channel dimension
    ball_input_size=train_ball_data[0].shape[1],
    hidden_size=64,
    num_layers=2,
    num_classes=1,
    dropout=0.5
).to(device)  # Move model to device

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # L2 regularization
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

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

# Plot training history using plot module
plot_training_history(
    epochs_range=epochs_range,
    train_losses=train_losses,
    val_losses=val_losses,
    train_accuracies=train_accuracies,
    val_accuracies=val_accuracies,
    save_path=os.path.join(save_dir, 'training_history.png')
)

# Log final plots to Weights & Biases
wandb.log({"training_history": wandb.Image(os.path.join(save_dir, 'training_history.png'))})

# Step 6: Evaluate Model on Test Data
model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
model.to(device)  # Move model to device
model.eval()
all_labels = []
all_predictions = []
all_probs = []

# Define window sizes
window_sizes = [20, 25, 30, 35, 40, 45]
stage_metrics = {}

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

    # Calculate metrics for each window size
    for window in window_sizes:
        window_length = window * 6  # Convert overs to balls
        if len(all_labels) >= window_length:
            window_labels = all_labels[:window_length]
            window_predictions = all_predictions[:window_length]
        else:
            window_labels = all_labels
            window_predictions = all_predictions

        accuracy = accuracy_score(window_labels, window_predictions)
        precision = precision_score(window_labels, window_predictions)
        recall = recall_score(window_labels, window_predictions)
        f1 = f1_score(window_labels, window_predictions)
        stage_metrics[f"{window} overs"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    overall_precision = precision_score(all_labels, all_predictions)
    overall_recall = recall_score(all_labels, all_predictions)
    overall_f1 = f1_score(all_labels, all_predictions)
    
    overall_metrics = {
        "accuracy": overall_accuracy,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1
    }
    
    metrics = {
        "stage_metrics": stage_metrics,
        "overall_metrics": overall_metrics
    }
    
    # Convert metrics to pandas DataFrames
    stage_df = pd.DataFrame(stage_metrics).T
    stage_df.index.name = 'Stage'
    stage_df.reset_index(inplace=True)
    
    overall_df = pd.DataFrame([overall_metrics], index=['Overall']).reset_index()
    overall_df.rename(columns={'index': 'Stage'}, inplace=True)
    
    # Print metrics in DataFrame format
    print("\nStage Metrics:")
    print(stage_df.to_string(index=False))
    
    print("\nOverall Metrics:")
    print(overall_df.to_string(index=False))
    
    # Convert DataFrames to wandb Tables
    stage_table = wandb.Table(dataframe=stage_df)
    overall_table = wandb.Table(dataframe=overall_df)
    
    # Log metrics tables to Weights & Biases
    wandb.log({
        "Stage Metrics": stage_table,
        "Overall Metrics": overall_table
    })

# Step 7: Generate Evaluation Metrics
conf_matrix = confusion_matrix(all_labels, all_predictions)
print('Confusion Matrix:')
print(conf_matrix)

class_report = classification_report(all_labels, all_predictions, target_names=['Loss', 'Won'])
print('Classification Report:')
print(class_report)

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve using plot module
plot_roc_curve(
    fpr=fpr,
    tpr=tpr,
    roc_auc=roc_auc,
    save_path=os.path.join(save_dir, 'roc_curve.png')
)

# Log ROC curve to Weights & Biases
wandb.log({"roc_curve": wandb.Image(os.path.join(save_dir, 'roc_curve.png'))})

# Convert confusion matrix to DataFrame for logging
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Loss', 'Actual Won'], columns=['Predicted Loss', 'Predicted Won'])

# Log evaluation metrics to Weights & Biases
wandb.log({
    "confusion_matrix": wandb.Table(dataframe=conf_matrix_df),
})

# Finish the Weights & Biases run
wandb.finish()