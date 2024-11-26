import torch
import numpy as np
import random
import logging
import wandb
import os
import matplotlib.pyplot as plt
from .architecture import EncoderDecoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def initialize_wandb(project_name: str = "T20I-CRICKET-WINNER-PREDICTION"):
    if not wandb.run:
        wandb.init(project=project_name)
    return wandb.config


def set_default_config_if_not_present(config):
    """
    Sets default configuration values if they are not provided.

    Args:
        config: The configuration object.
    """
    defaults = {
        "batch_size": 16,
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.5,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "num_epochs": 100,
        "enable_plots": False,  # Add flag to control plotting
    }
    for key, value in defaults.items():
        if key not in config:
            config[key] = value


def initialize_model(config, train_dataset, device):
    # Extract sample data from the dataset
    sample_team_input, sample_player_input, sample_ball_input, _ = train_dataset[0]

    # Get input sizes based on sample data
    team_input_size = sample_team_input.shape[0]
    player_input_size = sample_player_input.shape
    ball_input_size = sample_ball_input.shape[-1]
    print(f"Team input size: {team_input_size}")
    print(f"Player input size: {player_input_size}")
    print(f"Ball input size: {ball_input_size}")
    # Check the shape of the first ball data entry
    model = EncoderDecoderModel(
        team_input_size=team_input_size,
        player_input_channels=1,
        player_input_height=player_input_size[0],
        player_input_width=player_input_size[1],
        ball_input_size=ball_input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=1,
        dropout=config["dropout"],
    ).to(device)
    return model


def plot_training_history(
    epochs_range,
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    save_path=None,
):
    plt.figure(figsize=(12, 4))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Training Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label="Training Accuracy")
    plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.close()


from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score


def evaluate_model(model, dataloader, device, window_sizes, config, save_dir):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    stage_metrics = {}

    with torch.no_grad():
        correct = 0
        total = 0
        for team, player, ball, labels in dataloader:
            team, player, ball, labels = (
                team.to(device),
                player.to(device),
                ball.to(device),
                labels.to(device),
            )
            outputs = model(team, player, ball)
            probs = outputs.squeeze().cpu().numpy()
            if probs.ndim == 0:
                probs = probs.reshape(1)
            predicted = (outputs.data > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Accuracy: {:.2f} %".format(100 * correct / total))

        for window in window_sizes:
            window_length = window * 6
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
                "f1": f1,
            }

        overall_accuracy = accuracy_score(all_labels, all_predictions)
        overall_precision = precision_score(all_labels, all_predictions)
        overall_recall = recall_score(all_labels, all_predictions)
        overall_f1 = f1_score(all_labels, all_predictions)

        overall_metrics = {
            "accuracy": overall_accuracy,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
        }

        metrics = {"stage_metrics": stage_metrics, "overall_metrics": overall_metrics}

        # Conditionally plot ROC curve
        if config["enable_plots"]:
            from sklearn.metrics import roc_curve, auc

            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)
            plot_roc_curve(fpr, tpr, roc_auc, os.path.join(save_dir, "roc_curve.png"))
            # Log ROC curve to Weights & Biases
            wandb.log(
                {"roc_curve": wandb.Image(os.path.join(save_dir, "roc_curve.png"))}
            )

        return metrics, all_labels, all_predictions, all_probs



def train_and_evaluate(
    model,
    train_dataloader,
    val_dataloader,
    config,
    device,
    save_dir,
    save_full_model=True,
    patience=10,
):
    # Ensure wandb is initialized
    if not wandb.run:
        initialize_wandb()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    trigger_times = 0
    num_epochs = config["num_epochs"]
    patience = 10

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    from tqdm import tqdm

    best_model_path = None  # Initialize best_model_path

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

            # Save model checkpoint to a file
            best_model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
            if save_full_model:
                torch.save(model, best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)

        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

    # Save the best model to Weights & Biases as an artifact after training is done
    artifact_model = wandb.Artifact(
        f"best_model_val_loss_{best_val_loss:.4f}", type="model"
    )
    artifact_model.add_file(best_model_path, name="best_model.pth")
    logged_artifact = wandb.log_artifact(artifact_model)
    logged_artifact.wait()  # Wait for the artifact to finish logging
    os.remove(best_model_path)  # Clean up the temporary file

    # Conditionally plot training history
    if config["enable_plots"]:
        epochs_range = range(1, len(train_losses) + 1)
        plot_training_history(
            epochs_range,
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            os.path.join(save_dir, "training_history.png"),
        )
        # Log plots to Weights & Biases
        wandb.log(
            {
                "training_history": wandb.Image(
                    os.path.join(save_dir, "training_history.png")
                )
            }
        )


def export_model_to_onnx(model, export_path, input_shapes):
    """
    Exports the PyTorch model to ONNX format with dynamic axes for variable-length inputs.

    Args:
        model (torch.nn.Module): The trained model to export.
        export_path (str): The file path to save the ONNX model.
        input_shapes (tuple): Example input tensors (team_input, player_input, ball_input).
    """
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            input_shapes,
            export_path,
            export_params=True,
            opset_version=12,
            input_names=['team_input', 'player_input', 'ball_input'],
            output_names=['output'],
            dynamic_axes={
                'team_input': {0: 'batch_size'},
                'player_input': {0: 'batch_size'},
                'ball_input': {0: 'batch_size', 1: 'ball_length'},
                'output': {0: 'batch_size'},
            },
        )
    print(f"Model has been exported to {export_path}")
