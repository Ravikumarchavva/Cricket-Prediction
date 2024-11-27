import sys
import os
sys.path.append(os.path.join(os.getcwd(),"..",".."))

from utils.data_utils import collate_fn_with_padding, load_datasets, augument_data
from utils.model_utils import set_seed
from torch.utils.data import DataLoader

set_seed()
# Load the Datasets
train_dataset, val_dataset, test_dataset = load_datasets()

# Step 2: Augment Data
train_dataset, val_dataset, test_dataset = augument_data(train_dataset, val_dataset, test_dataset)

# Step 3: Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_with_padding)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_with_padding)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_with_padding)

from utils.model_utils import initialize_model, set_default_config_if_not_present, export_model_to_onnx
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
config= {}
set_default_config_if_not_present(config)
model = initialize_model(config, train_dataset,device=device)

# Move the model to the device
model.to(device)

# Set the model to evaluation mode
model.eval()

# Visualize the model architecture
from torchinfo import summary  # Replace torchsummary with torchinfo
from torchviz import make_dot  # Add import for torchviz

# Visualize the model architecture using torchinfo
summary(model, input_size=[(1, 13), (1, 22, 12), (1, 10)])

# Create a dummy input to visualize the graph
team_dummy = torch.randn(1, 13).to(device)
player_dummy = torch.randn(1, 22, 12).to(device)
ball_dummy = torch.randn(1, 214, 10).to(device)  # Example with dynamic length

# Forward pass to get the output
output = model(team_dummy, player_dummy, ball_dummy)

# Generate and save the model visualization
dot = make_dot(output, params=None)
dot.format = 'png'
dot.render('model_visualization')  # Saves as model_visualization.png

# Export the model to ONNX
export_path = os.path.join(os.getcwd(), 'model.onnx')
export_model_to_onnx(model, export_path, (team_dummy, player_dummy, ball_dummy))