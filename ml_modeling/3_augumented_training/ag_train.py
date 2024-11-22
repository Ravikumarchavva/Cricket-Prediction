import os
import sys
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd

# Initialize Weights & Biases
wandb.init(project="T20I-Cricket-Win-Prediction")

sys.path.append(os.path.join(os.getcwd(), '..'))
from model_utils import (
    set_seed, initialize_logging, initialize_wandb, set_default_config, load_datasets, augument_data,
    create_datasets, create_dataloaders, initialize_model, evaluate_model, train_and_evaluate
)

def main():
    set_seed()
    logger = initialize_logging()
    config = initialize_wandb()
    set_default_config(config)  # Set default configuration values

    train_dataset, val_dataset, test_dataset = load_datasets()
    train_data, val_data, test_data = augument_data(train_dataset, val_dataset, test_dataset)
    train_dataset, val_dataset, test_dataset = create_datasets(train_data, val_data, test_data)

    batch_size = config.batch_size
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = initialize_model(config, train_dataset, device)
    save_dir = os.path.dirname(os.path.abspath(__file__))

    train_and_evaluate(model, train_dataloader, val_dataloader, config, device, save_dir)
    
    # Define window sizes
    window_sizes = [20, 25, 30, 35, 40, 45]
    metrics, all_labels, all_predictions, all_probs = evaluate_model(model, test_dataloader, device, window_sizes)

    # Calculate metrics for each window size
    stage_metrics = metrics["stage_metrics"]
    overall_metrics = metrics["overall_metrics"]

    # Convert metrics to pandas DataFrames
    stage_df = pd.DataFrame(stage_metrics).T
    stage_df.index.name = 'Stage'
    stage_df.reset_index(inplace=True)

    overall_df = pd.DataFrame(overall_metrics, index=['Overall']).reset_index()
    overall_df.rename(columns={'index': 'Stage'}, inplace=True)

    # Print metrics in DataFrame format
    print("\nStage Metrics:")
    print(stage_df.to_string(index=False))

    print("\nOverall Metrics:")
    print(overall_df.to_string(index=False))

    # Convert DataFrames to wandb Tables
    stage_table = wandb.Table(data=stage_df)
    overall_table = wandb.Table(data=overall_df)

    # Log metrics tables to Weights & Biases
    wandb.log({
        "Stage Metrics": stage_table,
        "Overall Metrics": overall_table
    })

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
        "confusion_matrix": wandb.Table(dataframe=conf_matrix_df)
    })

    # Finish the Weights & Biases run
    wandb.finish()

if __name__ == "__main__":
    main()