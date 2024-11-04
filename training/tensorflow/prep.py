import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import os

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu,True)

import numpy as np
import tensorflow as tf
import polars as pl

# Load the data (assuming your data files are in the correct directory as specified)
directory = '/app/data/filteredData'
balltoball = pl.read_csv(os.path.join(directory, 'balltoball.csv'))
teamStats = pl.read_csv(os.path.join(directory, 'team12Stats.csv'))
playersStats = pl.read_csv(os.path.join(directory, 'playersStats.csv'))


partitions = teamStats.partition_by(['match_id', 'flip'])
partitions = np.array([partition.drop(['match_id','flip']).to_numpy() for partition in partitions])
tstf = tf.data.Dataset.from_tensor_slices(partitions)
partitions = playersStats.partition_by(['match_id', 'flip'])
partitions = np.array([partition.drop(['match_id','flip']).to_numpy() for partition in partitions])
pstf = tf.data.Dataset.from_tensor_slices(partitions)
partitions = balltoball.partition_by(['match_id', 'flip'])
# Create a ragged tensor from a list of tensors
ragged_tensor = tf.ragged.constant([partition.drop(['match_id','flip']).to_numpy() for partition in partitions])
bbtf = tf.data.Dataset.from_tensor_slices(ragged_tensor)


# Print the shapes of the datasets
for sample in tstf.take(1):
    print("Team Stats Sample Shape:", sample.shape)

for sample in pstf.take(1):
    print("Players Stats Sample Shape:", sample.shape)

for sample in bbtf.take(1):
    print("Ball to Ball Stats Sample Shape:", sample.shape)
    print("Ball to Ball Stats Sample Value:", sample[0])

combined_dataset = tf.data.Dataset.zip((tstf, pstf, bbtf))
for sample in combined_dataset.take(1):
    print("Team Stats Sample Shape:", sample[0].shape)
    print("Players Stats Sample Shape:", sample[1].shape)
    print("Ball to Ball Stats Sample Shape:", sample[2].shape)
    print("Sample 0:", sample)


# Assuming `combined_dataset` is your tf.data.Dataset containing the ball stats and labels
def extract_labels_and_data(combined_dataset):
    data_samples = []
    labels = []
    for sample in combined_dataset:
        # Convert ragged tensor to uniform tensor
        ball_stats = sample[2].to_tensor()
        # Extract ball stats and labels
        data_sample = ball_stats[:, :-1]  # Assuming last column is the label
        label = ball_stats[:, -1]  # Last column as labels
        data_samples.append(data_sample)
        labels.append(label)
    return data_samples, labels

ball_data_samples, labels = extract_labels_and_data(combined_dataset)
# Prepare the data for training
def prepare_dataset(combined_dataset):
    team_stats_data = []
    player_stats_data = []
    ball_stats_data = []
    labels = []
    for sample in combined_dataset:
        team_stats_sample = sample[0]
        player_stats_sample = sample[1]
        ball_stats_sample = sample[2].to_tensor()

        # Assuming last column is the label
        data_sample = ball_stats_sample[:, :-1]
        label = ball_stats_sample[0, -1]  # Assuming label is the same across the sequence

        team_stats_data.append(team_stats_sample)
        player_stats_data.append(player_stats_sample)
        ball_stats_data.append(data_sample)
        labels.append(label)

    # Pad ball_stats_data sequences to the same length
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    ball_stats_data = pad_sequences([data.numpy() for data in ball_stats_data], padding='post', dtype='float32')

    return (tf.stack(team_stats_data), tf.stack(player_stats_data), tf.convert_to_tensor(ball_stats_data)), tf.convert_to_tensor(labels)

# Prepare the dataset
inputs, labels = prepare_dataset(combined_dataset)

# save the prepared data
tf.data.experimental.save(inputs, '/app/data/reshaped/inputs')

# Adjust input shapes based on prepared data
team_input_shape = inputs[0].shape[1:]
player_input_shape = inputs[1].shape[1:]
team_input_shape, player_input_shape, inputs[2].shape, labels.shape
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import os

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu,True)

import numpy as np
import tensorflow as tf
import polars as pl

# Load the data (assuming your data files are in the correct directory as specified)
directory = '/app/data/filteredData'
balltoball = pl.read_csv(os.path.join(directory, 'balltoball.csv'))
teamStats = pl.read_csv(os.path.join(directory, 'team12Stats.csv'))
playersStats = pl.read_csv(os.path.join(directory, 'playersStats.csv'))


partitions = teamStats.partition_by(['match_id', 'flip'])
partitions = np.array([partition.drop(['match_id','flip']).to_numpy() for partition in partitions])
tstf = tf.data.Dataset.from_tensor_slices(partitions)
partitions = playersStats.partition_by(['match_id', 'flip'])
partitions = np.array([partition.drop(['match_id','flip']).to_numpy() for partition in partitions])
pstf = tf.data.Dataset.from_tensor_slices(partitions)
partitions = balltoball.partition_by(['match_id', 'flip'])
# Create a ragged tensor from a list of tensors
ragged_tensor = tf.ragged.constant([partition.drop(['match_id','flip']).to_numpy() for partition in partitions])
bbtf = tf.data.Dataset.from_tensor_slices(ragged_tensor)


# Print the shapes of the datasets
for sample in tstf.take(1):
    print("Team Stats Sample Shape:", sample.shape)

for sample in pstf.take(1):
    print("Players Stats Sample Shape:", sample.shape)

for sample in bbtf.take(1):
    print("Ball to Ball Stats Sample Shape:", sample.shape)
    print("Ball to Ball Stats Sample Value:", sample[0])

combined_dataset = tf.data.Dataset.zip((tstf, pstf, bbtf))
for sample in combined_dataset.take(1):
    print("Team Stats Sample Shape:", sample[0].shape)
    print("Players Stats Sample Shape:", sample[1].shape)
    print("Ball to Ball Stats Sample Shape:", sample[2].shape)
    print("Sample 0:", sample)

# Assuming `combined_dataset` is your tf.data.Dataset containing the ball stats and labels
def extract_labels_and_data(combined_dataset):
    data_samples = []
    labels = []
    for sample in combined_dataset:
        # Convert ragged tensor to uniform tensor
        ball_stats = sample[2].to_tensor()
        # Extract ball stats and labels
        data_sample = ball_stats[:, :-1]  # Assuming last column is the label
        label = ball_stats[:, -1]  # Last column as labels
        data_samples.append(data_sample)
        labels.append(label)
    return data_samples, labels

ball_data_samples, labels = extract_labels_and_data(combined_dataset)
# Prepare the data for training
def prepare_dataset(combined_dataset):
    team_stats_data = []
    player_stats_data = []
    ball_stats_data = []
    labels = []
    for sample in combined_dataset:
        team_stats_sample = sample[0]
        player_stats_sample = sample[1]
        ball_stats_sample = sample[2].to_tensor()

        # Assuming last column is the label
        data_sample = ball_stats_sample[:, :-1]
        label = ball_stats_sample[0, -1]  # Assuming label is the same across the sequence

        team_stats_data.append(team_stats_sample)
        player_stats_data.append(player_stats_sample)
        ball_stats_data.append(data_sample)
        labels.append(label)

    # Pad ball_stats_data sequences to the same length
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    ball_stats_data = pad_sequences([data.numpy() for data in ball_stats_data], padding='post', dtype='float32')

    return (tf.stack(team_stats_data), tf.stack(player_stats_data), tf.convert_to_tensor(ball_stats_data)), tf.convert_to_tensor(labels)

# Prepare the dataset
inputs, labels = prepare_dataset(combined_dataset)

# Adjust input shapes based on prepared data
team_input_shape = inputs[0].shape[1:]
player_input_shape = inputs[1].shape[1:]
print(team_input_shape, player_input_shape, inputs[2].shape, labels.shape)


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import layers
# Define the Team Stats Model (DNN)
class TeamStatsModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(TeamStatsModel, self).__init__()
        self.dense1 = layers.Dense(64, activation="relu", kernel_initializer="he_normal")
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        
        self.dense2 = layers.Dense(32, activation="relu", kernel_initializer="he_normal")
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)
        
        self.output_layer = layers.Dense(16, activation="relu", kernel_initializer="he_normal")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        return self.output_layer(x)

# Define the Player Stats Model (CNN)
class PlayerStatsModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(PlayerStatsModel, self).__init__()
        self.conv1 = layers.Conv1D(32, kernel_size=3, activation="relu", kernel_initializer="he_normal")
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling1D(pool_size=2)

        self.conv2 = layers.Conv1D(64, kernel_size=3, activation="relu", kernel_initializer="he_normal")
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling1D(pool_size=2)

        self.flatten = layers.Flatten()
        self.output_layer = layers.Dense(16, activation="relu", kernel_initializer="he_normal")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)  # Fixed from inputs to x
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        return self.output_layer(x)

# Update the BallToBallModel
class BallToBallModel(tf.keras.Model):
    def __init__(self):
        super(BallToBallModel, self).__init__()
        # Add a projection layer to match the input dimension to the model dimension
        self.projection = layers.Dense(128)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention1 = layers.MultiHeadAttention(num_heads=8, key_dim=128)
        self.dropout1 = layers.Dropout(0.3)
        self.ffn1 = tf.keras.Sequential([
            layers.Dense(128, activation='relu', kernel_initializer="he_normal"),
            layers.Dense(128, activation='relu', kernel_initializer="he_normal"),
        ])

        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.attention2 = layers.MultiHeadAttention(num_heads=8, key_dim=128)
        self.dropout2 = layers.Dropout(0.3)
        self.ffn2 = tf.keras.Sequential([
            layers.Dense(128, activation='relu', kernel_initializer="he_normal"),
            layers.Dense(128, activation='relu', kernel_initializer="he_normal"),
        ])

        self.layer_norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.attention3 = layers.MultiHeadAttention(num_heads=8, key_dim=128)
        self.dropout3 = layers.Dropout(0.3)
        self.ffn3 = tf.keras.Sequential([
            layers.Dense(128, activation='relu', kernel_initializer="he_normal"),
            layers.Dense(128, activation='relu', kernel_initializer="he_normal"),
        ])

        self.global_pool = layers.GlobalAveragePooling1D()
        self.mlp = tf.keras.Sequential([
            layers.Dense(256, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(16, activation="relu", kernel_initializer="he_normal"),
        ])

    def call(self, inputs):
        # Project inputs to match the model dimension
        x = self.projection(inputs)
        x = self.layer_norm1(x)
        attn_output = self.attention1(x, x)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output  # Residual connection

        ffn_output = self.ffn1(x)
        x = x + ffn_output  # Residual connection

        # Second transformer block
        x = self.layer_norm2(x)
        attn_output = self.attention2(x, x)
        attn_output = self.dropout2(attn_output)
        x = x + attn_output

        ffn_output = self.ffn2(x)
        x = x + ffn_output

        # Third transformer block
        x = self.layer_norm3(x)
        attn_output = self.attention3(x, x)
        attn_output = self.dropout3(attn_output)
        x = x + attn_output

        ffn_output = self.ffn3(x)
        x = x + ffn_output

        # Global pooling and final MLP
        x = self.global_pool(x)
        return self.mlp(x)

# Update the CombinedModel to reflect the change
class CombinedModel(tf.keras.Model):
    def __init__(self, team_input_shape, player_input_shape):
        super(CombinedModel, self).__init__()
        self.team_model = TeamStatsModel(team_input_shape)
        self.player_model = PlayerStatsModel(player_input_shape)
        self.ball_model = BallToBallModel()
        
        self.final_mlp1 = layers.Dense(64, activation="relu", kernel_initializer="he_normal")
        self.dropout = layers.Dropout(0.3)
        self.final_mlp2 = layers.Dense(32, activation="relu", kernel_initializer="he_normal")
        self.final_output = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        team_input, player_input, ball_input = inputs
        team_output = self.team_model(team_input)        # Shape: (batch_size, 1, 16)
        team_output = layers.Flatten()(team_output)      # Shape: (batch_size, 16)
        player_output = self.player_model(player_input)  # Shape: (batch_size, 16)
        ball_output = self.ball_model(ball_input)        # Shape: (batch_size, 16)
        
        # Concatenate along the last axis
        combined = layers.concatenate([team_output, player_output, ball_output], axis=-1)
        x = self.final_mlp1(combined)
        x = self.dropout(x)
        x = self.final_mlp2(x)
        return self.final_output(x)

# Import callbacks for learning rate scheduling and early stopping
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

# Check the shapes of inputs and labels
team_input, player_input, ball_input = inputs
print(f"Shape of team_input: {team_input.shape}")
print(f"Shape of player_input: {player_input.shape}")
print(f"Shape of ball_input: {ball_input.shape}")
print(f"Shape of labels: {labels.shape}")

# Ensure inputs and labels have the same number of samples
min_samples = min(team_input.shape[0], player_input.shape[0], ball_input.shape[0], labels.shape[0])
team_input = team_input[:min_samples]
player_input = player_input[:min_samples]
ball_input = ball_input[:min_samples]
labels = labels[:min_samples]

# Ensure inputs have the same shape
team_input = tf.reshape(team_input, (min_samples, *team_input_shape))
player_input = tf.reshape(player_input, (min_samples, *player_input_shape))
ball_input = tf.reshape(ball_input, (min_samples, *inputs[2].shape[1:]))

# Split the data into training and validation sets
(train_team_input, val_team_input, train_player_input, val_player_input, train_ball_input, val_ball_input, train_labels, val_labels) = train_test_split(
    team_input.numpy(), player_input.numpy(), ball_input.numpy(), labels.numpy(), test_size=0.2, random_state=42)

# Convert the split data back to tensors
train_team_input = tf.convert_to_tensor(train_team_input)
val_team_input = tf.convert_to_tensor(val_team_input)
train_player_input = tf.convert_to_tensor(train_player_input)
val_player_input = tf.convert_to_tensor(val_player_input)
train_ball_input = tf.convert_to_tensor(train_ball_input)
val_ball_input = tf.convert_to_tensor(val_ball_input)
train_labels = tf.convert_to_tensor(train_labels)
val_labels = tf.convert_to_tensor(val_labels)

# Combine the inputs for training and validation
train_inputs = [train_team_input, train_player_input, train_ball_input]
val_inputs = [val_team_input, val_player_input, val_ball_input]

# Instantiate and compile the model with a learning rate scheduler
model = CombinedModel(team_input_shape, player_input_shape)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Define callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Add callbacks for early stopping and model checkpointing
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)

# Train the model with updated callbacks
history = model.fit(
    train_inputs, train_labels,
    epochs=200,
    batch_size=16,
    validation_data=(val_inputs, val_labels),
    callbacks=[early_stopping, lr_scheduler, model_checkpoint]
)

# Evaluate the model
loss, accuracy = model.evaluate(val_inputs, val_labels)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")