import os
import sys
import torch
import wandb
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd

# Initialize Weights & Biases
wandb.init(project="T20I-CRICKET-WINNER-PREDICTION")

sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from utils.model_utils import (
    set_seed,
    initialize_logging,
    initialize_wandb,
    plot_roc_curve,
    initialize_model,
    set_default_config_if_not_present,
    train_and_evaluate,
)
from utils.data_utils import load_datasets, create_dataloaders


def evaluate_model(model, test_dataloader, device, save_dir=os.getcwd()):
    model.load_state_dict(
        torch.load(os.path.join(save_dir, "best_model.pth"), weights_only=True)
    )
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    with torch.no_grad():
        correct = 0
        total = 0
        for team, player, ball, labels in test_dataloader:
            team, player, ball, labels = (
                team.to(device),
                player.to(device),
                ball.to(device),
                labels.to(device),
            )
            labels = labels.float()

            outputs = model(team, player, ball)
            probs = outputs.squeeze().cpu().numpy()
            predicted = (outputs.data > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Test Accuracy: {:.2f} %".format(100 * correct / total))
    return all_labels, all_predictions, all_probs


def main():
    set_seed()
    logger = initialize_logging()

    # Ensure wandb is initialized
    if not wandb.run:
        config = initialize_wandb()
    else:
        config = wandb.config

    set_default_config_if_not_present(config)  # Set default configuration values

    # Load datasets only once
    train_dataset, val_dataset, test_dataset = load_datasets()

    batch_size = config.batch_size
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = initialize_model(config, train_dataset, device)
    save_dir = os.path.dirname(os.path.abspath(__file__))

    train_and_evaluate(
        model, train_dataloader, val_dataloader, config, device, save_dir
    )

    # Evaluate the model on the test set
    all_labels, all_predictions, all_probs = evaluate_model(
        model, test_dataloader, device
    )

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

    plot_roc_curve(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

    # Convert confusion matrix to DataFrame for logging
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=["Actual Class 0", "Actual Class 1"],
        columns=["Predicted Class 0", "Predicted Class 1"],
    )

    # Log evaluation metrics to Weights & Biases
    wandb.log({"confusion_matrix": wandb.Table(dataframe=conf_matrix_df)})

    # Finish the Weights & Biases run
    wandb.finish()


if __name__ == "__main__":
    main()
