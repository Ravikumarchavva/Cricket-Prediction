import wandb
import os
import torch

# Initialize the W&B API
api = wandb.Api()

# Access the run and the artifact you want to download
artifact = api.artifact('ravikumarchavva-org/T20I-CRICKET-WINNER-PREDICTION/best_model_val_loss_0.3673:v0', type='model')
config = artifact.metadata
model_path = artifact.download()

import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))

from data_utils import collate_fn_with_padding, load_datasets, augument_data, dataset_to_list
from torch.utils.data import DataLoader

# Load the Datasets
train_dataset, test_dataset, val_dataset = load_datasets()

# Step 2: Augment Data
train_dataset, test_dataset, val_dataset = augument_data(train_dataset, test_dataset, val_dataset)

# Step 4: Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn_with_padding)
val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn_with_padding)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn_with_padding)

# Step 5: Extract Data
train_team_data, train_player_data, train_ball_data, train_labels = dataset_to_list(train_dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os

# Load the model
model = torch.load(os.path.join(model_path, 'best_model.pth'),weights_only=False)

from model_utils import evaluate_model, plot_roc_curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

config['enable_plots'] = False

save_dir = os.path.dirname(os.getcwd())

# Define window sizes
window_sizes = [20, 25, 30, 35, 40, 45]
print(config)
metrics, all_labels, all_predictions, all_probs = evaluate_model(
    model, test_dataloader, device, window_sizes, config=config, save_dir=save_dir
)

# Calculate metrics for each window size
stage_metrics = metrics["stage_metrics"]
overall_metrics = metrics["overall_metrics"]

import pandas as pd
# Convert metrics to pandas DataFrames
stage_df = pd.DataFrame(stage_metrics).T
stage_df.index.name = "Stage"
stage_df.reset_index(inplace=True)

overall_df = pd.DataFrame(overall_metrics, index=["Overall"]).reset_index()
overall_df.rename(columns={"index": "Stage"}, inplace=True)

# Print metrics in DataFrame format
print("\nStage Metrics:")
print(stage_df.to_string(index=False))

print("\nOverall Metrics:")
print(overall_df.to_string(index=False))

# Convert DataFrames to wandb Tables
stage_table = wandb.Table(data=stage_df)
overall_table = wandb.Table(data=overall_df)

# Log metrics tables to Weights & Biases
wandb.log({"Stage Metrics": stage_table, "Overall Metrics": overall_table})

# Step 7: Generate Evaluation Metrics
conf_matrix = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(
    all_labels, all_predictions, target_names=["Class 0", "Class 1"]
)
print("Classification Report:")
print(class_report)

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plot_roc_curve(fpr=fpr, tpr=tpr, roc_auc=roc_auc, save_path=save_dir)

# Convert confusion matrix to DataFrame for logging
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=["Actual Class 0", "Actual Class 1"],
    columns=["Predicted Class 0", "Predicted Class 1"],
)