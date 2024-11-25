import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.model_utils import (
    set_seed, initialize_logging,
    initialize_model, train_and_evaluate, evaluate_model, set_default_config_if_not_present
)
from utils.data_utils import create_dataloaders, load_datasets, augument_data

def train(config, train_dataset, val_dataset, test_dataset, device):
    set_seed()
    logger = initialize_logging()
    set_default_config_if_not_present(config)

    batch_size = config.batch_size
    train_dataset, val_dataset, test_dataset = augument_data(
        train_dataset, val_dataset, test_dataset
    )
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size)

    logger.info(f"Using device: {device}")
    model = initialize_model(config, train_dataset, device)
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    train_and_evaluate(model, train_dataloader, val_dataloader, config, device, save_dir)
    window_sizes = [20, 25, 30, 35, 40, 45]
    metrics, all_labels, all_predictions, all_probs = evaluate_model(
        model, test_dataloader, device, window_sizes, config, save_dir=os.getcwd()
    )
    return metrics, all_labels, all_predictions, all_probs

if __name__ == "__main__":
    # Example usage
    config={}
    set_default_config_if_not_present(config)
    train_dataset, val_dataset, test_dataset = load_datasets()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(config, train_dataset, val_dataset, test_dataset, device)