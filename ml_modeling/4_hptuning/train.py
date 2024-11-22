import os
import sys
import torch
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_utils import (
    set_seed, initialize_logging, initialize_wandb, load_datasets, augument_data,
    create_datasets, create_dataloaders, initialize_model, train_and_evaluate, evaluate_model, set_default_config
)

def main():
    set_seed()
    logger = initialize_logging()
    config = initialize_wandb()
    set_default_config(config)


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
    window_sizes = [20,25,30,35,40,45]
    metrics, all_labels, all_predictions, all_probs = evaluate_model(
        model, test_dataloader, device, window_sizes, config, save_dir=os.getcwd()
    )

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