import wandb

# Initialize the W&B API
api = wandb.Api()

# Access the run and the artifact you want to download
artifact = api.artifact('ravikumarchavva-org/Cricket-Win-Prediction/best_model:v64')

# # Download the artifact
artifact.download('best_model.h5')
