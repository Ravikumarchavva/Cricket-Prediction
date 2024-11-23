import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import polars as pl
from typing import List, Tuple
import random
import logging
import wandb
import os
import matplotlib.pyplot as plt
from architecture import EncoderDecoderModel, CricketDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn_with_padding(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to pad sequences and stack them into a batch.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]): List of tuples containing team, player, ball data, and labels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Padded and stacked team, player, ball data, and labels.
    """
    team_batch, player_batch, ball_batch, labels = zip(*batch)
    team_batch = [
        team.clone().detach() if torch.is_tensor(team) else torch.tensor(team)
        for team in team_batch
    ]
    player_batch = [
        player.clone().detach() if torch.is_tensor(player) else torch.tensor(player)
        for player in player_batch
    ]
    ball_batch = [
        ball.clone().detach() if torch.is_tensor(ball) else torch.tensor(ball)
        for ball in ball_batch
    ]

    team_batch = pad_sequence(team_batch, batch_first=True, padding_value=0)
    player_batch = pad_sequence(player_batch, batch_first=True, padding_value=0)
    ball_batch = pad_sequence(ball_batch, batch_first=True, padding_value=0)

    labels = torch.tensor(labels).float().unsqueeze(1)
    return team_batch, player_batch, ball_batch, labels


def partition_data_with_keys(
    df: pl.DataFrame, group_keys: List[str]
) -> Tuple[List[Tuple], List[np.ndarray]]:
    """
    Partitions the dataframe by group keys and returns the keys and partitions.

    Args:
        df: Dataframe to be partitioned.
        group_keys: Keys to group by.

    Returns:
        Tuple[List[Tuple], List[np.ndarray]]: List of unique keys and list of partitions.
    """
    partitions = df.partition_by(group_keys)
    keys = [
        tuple(partition.select(group_keys).unique().to_numpy()[0])
        for partition in partitions
    ]
    partitions = [partition.drop(group_keys).to_numpy() for partition in partitions]
    return keys, partitions


def extract_data(
    dataset,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Extracts data from the dataset.

    Args:
        dataset: Dataset to extract data from.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]: Extracted team, player, ball data, and labels.
    """
    team_stats_list = []
    player_stats_list = []
    ball_stats_list = []
    labels_list = []

    for data in dataset:
        team_input, player_input, ball_input, label = data
        team_stats_list.append(team_input.numpy())
        player_stats_list.append(player_input.numpy())
        ball_stats_list.append(ball_input.numpy())
        labels_list.append(label.item())

    return team_stats_list, player_stats_list, ball_stats_list, labels_list


def augument_function(
    balltoball: List[np.ndarray],
    team_stats: List[np.ndarray],
    player_stats: List[np.ndarray],
    labels: List[int],
    window_sizes: List[int] = [20, 25, 30, 35, 40, 45],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Augments data to simulate predictions at various match stages using sliding windows.

    Args:
        balltoball (List[np.ndarray]): List of ball-by-ball data arrays.
        team_stats (List[np.ndarray]): List of team statistics arrays.
        player_stats (List[np.ndarray]): List of player statistics arrays.
        labels (List[int]): List of match outcome labels.
        window_sizes (List[int]): List of window sizes (in overs).

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]: Augmented balltoball, team_stats, player_stats, and labels.
    """
    augmented_balltoball = []
    augmented_team_stats = []
    augmented_player_stats = []
    augmented_labels = []

    for i in range(len(balltoball)):
        match_length = balltoball[i].shape[0]  # Total number of balls in the match
        for window in window_sizes:
            window_length = window * 6  # Convert overs to balls
            if match_length >= window_length:
                # Slice the data
                augmented_balltoball.append(balltoball[i][:window_length])
                augmented_team_stats.append(team_stats[i])
                augmented_player_stats.append(player_stats[i])
                augmented_labels.append(labels[i])  # Same label for augmented samples
            else:
                augmented_balltoball.append(balltoball[i])
                augmented_team_stats.append(team_stats[i])
                augmented_player_stats.append(player_stats[i])
                augmented_labels.append(labels[i])

    return (
        augmented_balltoball,
        augmented_team_stats,
        augmented_player_stats,
        augmented_labels,
    )


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


def set_default_config(config):
    """
    Sets default configuration values if they are not provided.

    Args:
        config: The configuration object.
    """
    defaults = {
        "batch_size": 32,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.5,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "num_epochs": 50,
        "enable_plots": False,  # Add flag to control plotting
    }
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)


import pickle


def load_datasets(
    entity_name: str = "ravikumarchavva-org",
    project: str = "T20I-CRICKET-WINNER-PREDICTION",
    dataset_name: str = "cricket-dataset",
    version: str = "latest",
) -> Tuple[Dataset, Dataset, Dataset]:
    # Ensure wandb is initialized
    if not wandb.run:
        wandb.init(project="T20I-CRICKET-WINNER-PREDICTION")

    # Use the artifact
    artifact = wandb.run.use_artifact(
        f"{entity_name}/{project}/{dataset_name}:{version}"
    )

    # Load datasets directly into memory and remove files after loading
    train_path = artifact.get_entry("train_dataset.pkl").download()
    with open(train_path, "rb") as train_file:
        train_dataset = pickle.load(train_file)
    os.remove(train_path)  # Clean up

    val_path = artifact.get_entry("val_dataset.pkl").download()
    with open(val_path, "rb") as val_file:
        val_dataset = pickle.load(val_file)
    os.remove(val_path)  # Clean up

    test_path = artifact.get_entry("test_dataset.pkl").download()
    with open(test_path, "rb") as test_file:
        test_dataset = pickle.load(test_file)
    os.remove(test_path)  # Clean up

    # Log dataset details
    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Validation Dataset: {len(val_dataset)} samples")
    print(f"Test Dataset: {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset


def augument_data(
    train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Augment the data by creating multiple samples for each match at different stages

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Augmented training, validation, and test datasets
    """

    train_team_data, train_player_data, train_ball_data, train_labels = extract_data(
        train_dataset
    )
    val_team_data, val_player_data, val_ball_data, val_labels = extract_data(
        val_dataset
    )
    test_team_data, test_player_data, test_ball_data, test_labels = extract_data(
        test_dataset
    )

    train_ball_data, train_team_data, train_player_data, train_labels = (
        augument_function(
            train_ball_data, train_team_data, train_player_data, train_labels
        )
    )
    val_ball_data, val_team_data, val_player_data, val_labels = augument_function(
        val_ball_data, val_team_data, val_player_data, val_labels
    )
    test_ball_data, test_team_data, test_player_data, test_labels = augument_function(
        test_ball_data, test_team_data, test_player_data, test_labels
    )

    train_dataset = CricketDataset(
        train_team_data, train_player_data, train_ball_data, train_labels
    )
    val_dataset = CricketDataset(
        val_team_data, val_player_data, val_ball_data, val_labels
    )
    test_dataset = CricketDataset(
        test_team_data, test_player_data, test_ball_data, test_labels
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


def create_datasets(train_data, val_data, test_data):
    train_dataset = CricketDataset(*train_data)
    val_dataset = CricketDataset(*val_data)
    test_dataset = CricketDataset(*test_data)
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_padding,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_padding,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_padding,
    )
    return train_dataloader, val_dataloader, test_dataloader


def initialize_model(config, train_dataset, device):
    train_team_data, _, train_ball_data, _ = extract_data(train_dataset)
    model = EncoderDecoderModel(
        team_input_size=train_team_data[0].shape[0],
        player_input_channels=1,
        ball_input_size=train_ball_data[0].shape[1],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=1,
        dropout=config['dropout'],
    ).to(device)
    return model


def plot_training_history(
    epochs_range, train_losses, val_losses, train_accuracies, val_accuracies, save_path
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
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
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

        print("Test Accuracy: {:.2f} %".format(100 * correct / total))

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
        if config.enable_plots:
            from sklearn.metrics import roc_curve, auc

            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)
            plot_roc_curve(fpr, tpr, roc_auc, os.path.join(save_dir, "roc_curve.png"))
            # Log ROC curve to Weights & Biases
            wandb.log(
                {"roc_curve": wandb.Image(os.path.join(save_dir, "roc_curve.png"))}
            )

        return metrics, all_labels, all_predictions, all_probs


import tempfile


def train_and_evaluate(
    model, train_dataloader, val_dataloader, config, device, save_dir
):
    # Ensure wandb is initialized
    if not wandb.run:
        initialize_wandb()

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

    from tqdm import tqdm

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

            # Save model checkpoint to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
                torch.save(model.state_dict(), tmp_file.name)
            best_model_path = tmp_file.name

        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

    # Save the best model to Weights & Biases as an artifact after training is done
    artifact_model = wandb.Artifact(f"best_model_val_loss_{best_val_loss:.4f}", type="model")
    artifact_model.add_file(best_model_path, "best_model.pth")

    # Log model metadata    
    artifact_model.metadata = {
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "device": device,
    }
    wandb.log_artifact(artifact_model)
    os.remove(best_model_path)  # Clean up the temporary file

    # Conditionally plot training history
    if config.enable_plots:
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
