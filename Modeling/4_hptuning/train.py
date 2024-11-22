import os
import sys
import torch
import torch.nn as nn
import wandb
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_utils import (
    set_seed, initialize_logging, initialize_wandb, load_datasets, preprocess_data,
    create_datasets, create_dataloaders, initialize_model, plot_roc_curve, evaluate_model
)

def train_and_evaluate(model, train_dataloader, val_dataloader, config, device, save_dir):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    wandb.watch(model, log="all", log_freq=100)

    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0
    num_epochs = 100

    def get_serializable_config(config):
        serializable_config = {}
        for k, v in config.__dict__.items():
            try:
                json.dumps(v)
                serializable_config[k] = v
            except (TypeError, OverflowError):
                continue
        return serializable_config

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for team, player, ball, labels in tqdm(train_dataloader):
            team, player, ball, labels = team.to(device), player.to(device), ball.to(device), labels.to(device)
            labels = labels.float()
            outputs = model(team, player, ball)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_dataloader)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for team, player, ball, labels in val_dataloader:
                team, player, ball, labels = team.to(device), player.to(device), ball.to(device), labels.to(device)
                labels = labels.float()
                outputs = model(team, player, ball)
                val_loss += criterion(outputs, labels).item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_dataloader)

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
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            serializable_config = get_serializable_config(config)
            with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
                json.dump(serializable_config, f)
            artifact = wandb.Artifact('best_model', type='model')
            artifact.add_file(os.path.join(save_dir, 'best_model.pth'))
            artifact.add_file(os.path.join(save_dir, 'model_config.json'))
            wandb.log_artifact(artifact)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

def evaluate_model_with_config(model, test_dataloader, device, save_dir):
    with open(os.path.join(save_dir, 'model_config.json'), 'r') as f:
        config_dict = json.load(f)
    config = type('Config', (object,), config_dict)

    # Extract input sizes from the test data
    sample_batch = next(iter(test_dataloader))
    team_sample, player_sample, ball_sample, _ = sample_batch
    team_input_size = team_sample.shape[1]
    player_input_channels = 1  # Assuming player data has a single channel
    ball_input_size = ball_sample.shape[2]

    model = initialize_model(config, team_input_size, ball_input_size, device)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        correct = 0
        total = 0
        for team, player, ball, labels in test_dataloader:
            team, player, ball, labels = team.to(device), player.to(device), ball.to(device), labels.to(device)
            team = team.float()
            player = player.float()
            ball = ball.float()
            labels = labels.float()

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

        print('Test Accuracy: {:.2f} %'.format(100 * correct / total))

        overall_metrics = {
            "accuracy": accuracy_score(all_labels, all_predictions),
            "precision": precision_score(all_labels, all_predictions),
            "recall": recall_score(all_labels, all_predictions),
            "f1": f1_score(all_labels, all_predictions)
        }

        wandb.log({
            "Overall Accuracy": overall_metrics["accuracy"],
            "Overall Precision": overall_metrics["precision"],
            "Overall Recall": overall_metrics["recall"],
            "Overall F1-Score": overall_metrics["f1"]
        })

        model_artifact = wandb.Artifact(
            name='best_model',
            type='model',
            metadata=overall_metrics
        )
        model_artifact.add_file(os.path.join(save_dir, 'best_model.pth'))
        wandb.log_artifact(model_artifact)

        conf_matrix = confusion_matrix(all_labels, all_predictions)
        print('Confusion Matrix:')
        print(conf_matrix)

        class_report = classification_report(all_labels, all_predictions, target_names=['Loss', 'Won'])
        print('Classification Report:')
        print(class_report)

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        plot_roc_curve(
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc,
            save_path=os.path.join(save_dir, 'roc_curve.png')
        )

        wandb.log({"roc_curve": wandb.Image(os.path.join(save_dir, 'roc_curve.png'))})
        os.remove(os.path.join(save_dir, 'roc_curve.png'))

def main():
    set_seed()
    logger = initialize_logging()
    config = initialize_wandb()

    train_dataset, val_dataset, test_dataset = load_datasets()
    train_data, val_data, test_data = preprocess_data(train_dataset, val_dataset, test_dataset)
    train_dataset, val_dataset, test_dataset = create_datasets(train_data, val_data, test_data)

    batch_size = config.batch_size
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = initialize_model(config, train_data[0], train_data[2], device)
    save_dir = os.path.dirname(os.path.abspath(__file__))

    train_and_evaluate(model, train_dataloader, val_dataloader, config, device, save_dir)
    evaluate_model_with_config(model, test_dataloader, device, save_dir)

    wandb.config.update({
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "save_dir": save_dir
    })

    wandb.finish()

if __name__ == "__main__":
    main()