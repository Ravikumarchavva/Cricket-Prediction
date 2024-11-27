import wandb
import torch
import os
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def fetch_best_run(api, entity, project, sweep_id, metric_name="val_accuracy", maximize=True):
    # Fetch the sweep
    print(f"Fetching sweep: {entity}/{project}/{sweep_id}")
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # Retrieve all runs in the sweep
    runs = sweep.runs

    # Sort runs by the specified metric
    best_run = sorted(
        runs,
        key=lambda run: run.summary.get(metric_name, float("-inf" if maximize else "inf")),
        reverse=maximize
    )[0]

    print(f"Best run ID: {best_run.id}")
    print(f"{metric_name}: {best_run.summary.get(metric_name)}")
    print(f"Config: {best_run.config}")

    return best_run

def download_best_model(run, artifact_name_prefix="best_model"):
    artifact_dir = None
    for artifact in run.logged_artifacts():
        if artifact.name.startswith(artifact_name_prefix):
            print(f"Downloading artifact: {artifact.name}")
            try:
                artifact_dir = artifact.download()
                print(f"Artifact downloaded to: {artifact_dir}")
            except Exception as e:
                print(f"Error downloading artifact {artifact.name}: {e}")
    return artifact_dir

def load_model(artifact_directory, model_filename="best_model.pth"):
    model_path = os.path.join(artifact_directory, model_filename)
    model = torch.load(model_path, weights_only=False)
    return model

def prepare_dataloaders(config):
    # Update sys.path to import utils
    sys.path.append(os.path.join(os.getcwd(), "..", ".."))

    from utils.data_utils import (
        collate_fn_with_padding,
        load_datasets,
        augument_data,
    )
    from torch.utils.data import DataLoader
    from utils.model_utils import set_seed

    set_seed()
    # Load the datasets
    train_dataset, val_dataset, test_dataset = load_datasets()

    # Augment data
    train_dataset, test_dataset, val_dataset = augument_data(
        train_dataset, test_dataset, val_dataset
    )

    # Create DataLoaders
    batch_size = config.get("batch_size", 32)
    collate_fn = collate_fn_with_padding

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader, test_dataloader

def evaluate_model_and_print_metrics(model, test_dataloader, device, window_sizes, config, save_dir):
    from utils.model_utils import evaluate_model, plot_roc_curve

    metrics, all_labels, all_predictions, all_probs = evaluate_model(
        model, test_dataloader, device, window_sizes, config, save_dir=save_dir
    )
    overall_metrics = metrics["overall_metrics"]
    stage_metrics = metrics["stage_metrics"]

    # Convert metrics to pandas DataFrames
    stage_df = pd.DataFrame(stage_metrics).T
    stage_df.index.name = "Stage"
    stage_df.reset_index(inplace=True)

    overall_df = pd.DataFrame(overall_metrics, index=["Overall"]).reset_index()
    overall_df.rename(columns={"index": "Stage"}, inplace=True)

    # Print metrics in DataFrame format
    print("\nOverall Metrics:")
    print(overall_df.to_string(index=False))

    print("\nStage Metrics:")
    print(stage_df)

    # Generate evaluation metrics
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    class_report = classification_report(
        all_labels, all_predictions, target_names=["Loss", "Win"]
    )
    print("Classification Report:")
    print(class_report)

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plot_roc_curve(fpr=fpr, tpr=tpr, roc_auc=roc_auc, save_path=save_dir)

    # Convert confusion matrix to DataFrame for logging
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=["Actual Lost", "Actual Win"],
        columns=["Predicted Lost", "Predicted Win"],
    )

    # Log the confusion matrix and classification report
    print("\nConfusion Matrix:")
    print(conf_matrix_df)

def main():
    # Initialize your WandB API
    api = wandb.Api()

    # Define your project and sweep details
    entity = "ravikumarchavva-org"
    project = "T20I-CRICKET-WINNER-PREDICTION"
    sweep_id = "qqakx1g3"
    metric_name = "val_accuracy"

    best_run = fetch_best_run(api, entity, project, sweep_id, metric_name)

    run_path = f"{entity}/{project}/{best_run.id}"
    run = api.run(run_path)

    artifact_dir = download_best_model(run)

    if artifact_dir is None:
        print("No artifact downloaded. Exiting.")
        return

    config = best_run.config

    train_dataset, test_dataloader = prepare_dataloaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model
    model = load_model(artifact_dir)

    # Evaluate the model
    config["enable_plots"] = False
    save_dir = os.getcwd()

    window_sizes = [20, 25, 30, 35, 40, 45]
    evaluate_model_and_print_metrics(
        model, test_dataloader, device, window_sizes, config, save_dir
    )

    wandb.finish()

if __name__ == "__main__":
    main()
