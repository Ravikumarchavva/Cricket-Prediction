import os
import sys
import torch
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd

# Initialize Weights & Biases
wandb.init(project="T20I-Cricket-Win-Prediction")

sys.path.append(os.path.join(os.getcwd(), ".."))
from model_utils import (
    set_seed,
    initialize_logging,
    initialize_wandb,
    load_datasets,
    preprocess_data,
    plot_roc_curve,
    create_datasets,
    create_dataloaders,
    initialize_model,
    evaluate_model,
    set_default_config,
)


def train_and_evaluate(
    model, train_dataloader, val_dataloader, config, device, save_dir
):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, verbose=True
    )

    best_val_loss = float("inf")
    patience = 10
    trigger_times = 0
    num_epochs = config.num_epochs

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
            team, player, ball, labels = (
                team.to(device),
                player.to(device),
                ball.to(device),
                labels.to(device),
            )
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
                team, player, ball, labels = (
                    team.to(device),
                    player.to(device),
                    ball.to(device),
                    labels.to(device),
                )
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

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Log metrics to Weights & Biases
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_accuracy": train_acc,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc,
            }
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            # Save model checkpoint to Weights & Biases
            wandb.save(os.path.join(save_dir, "best_model.pth"))
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

    # Log final plots to Weights & Biases
    wandb.log(
        {
            "training_history": wandb.Image(
                os.path.join(save_dir, "training_history.png")
            )
        }
    )


def main():
    set_seed()
    logger = initialize_logging()
    config = initialize_wandb()
    set_default_config(config)  # Set default configuration values

    train_dataset, val_dataset, test_dataset = load_datasets()
    train_data, val_data, test_data = preprocess_data(
        train_dataset, val_dataset, test_dataset
    )
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_data, val_data, test_data
    )

    batch_size = config.batch_size
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = initialize_model(config, train_data[0], train_data[2], device)
    save_dir = os.path.dirname(os.path.abspath(__file__))

    train_and_evaluate(
        model, train_dataloader, val_dataloader, config, device, save_dir
    )

    # Define window sizes
    window_sizes = [20, 25, 30, 35, 40, 45]
    metrics, all_labels, all_predictions, all_probs = evaluate_model(
        model, test_dataloader, device, window_sizes
    )

    # Calculate metrics for each window size
    stage_metrics = metrics["stage_metrics"]
    overall_metrics = metrics["overall_metrics"]

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

    # Log evaluation metrics to Weights & Biases
    wandb.log({"confusion_matrix": wandb.Table(dataframe=conf_matrix_df)})

    # Finish the Weights & Biases run
    wandb.finish()


if __name__ == "__main__":
    main()
