import wandb
import json

# Log in to your wandb account
# wandb.login()

api = wandb.Api()

# Fetch the sweep
sweep_id = "T20I/emvzlnnn"
sweep = api.sweep(sweep_id)

# Get all runs in the sweep
runs = sweep.runs
print(f"Found {len(runs)} runs in the sweep.")

for run in runs:
    # save the metrics for the run to a csv file
    metrics_dataframe = run.history()
    metrics_dataframe.to_csv("metrics.csv")
    
    # Extract and download JSON file
    json_artifact_path = metrics_dataframe['Stage Metrics'].dropna().values[0]['artifact_path']
    artifact = api.artifact(json_artifact_path)
    artifact.download()

    # Print JSON artifact path to view
    print(f"Downloaded JSON artifact from: {json_artifact_path}")