import torch
from torch.utils.data import Dataset

# Create a custom Dataset
class CricketDataset(Dataset):
    def __init__(self, team_stats_list, player_stats_list, ball_stats_list, labels):
        self.team_stats_list = team_stats_list
        self.player_stats_list = player_stats_list
        self.ball_stats_list = ball_stats_list
        self.labels = labels

    def __len__(self):
        return len(self.team_stats_list)

    def __getitem__(self, idx):
        team_input = torch.tensor(self.team_stats_list[idx], dtype=torch.float32)
        team_input = team_input.squeeze()
        player_input = torch.tensor(self.player_stats_list[idx], dtype=torch.float32)
        ball_stats = torch.tensor(self.ball_stats_list[idx], dtype=torch.float32)
        ball_input = ball_stats
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return team_input, player_input, ball_input, label

# Update the collate function to avoid UserWarning
from torch.nn.utils.rnn import pad_sequence

def collate_fn_with_padding(batch):
    team_batch, player_batch, ball_batch, labels = zip(*batch)
    team_batch = [team.clone().detach() if torch.is_tensor(team) else torch.tensor(team) for team in team_batch]
    player_batch = [player.clone().detach() if torch.is_tensor(player) else torch.tensor(player) for player in player_batch]
    ball_batch = [ball.clone().detach() if torch.is_tensor(ball) else torch.tensor(ball) for ball in ball_batch]

    team_batch = pad_sequence(team_batch, batch_first=True, padding_value=0)
    player_batch = pad_sequence(player_batch, batch_first=True, padding_value=0)
    ball_batch = pad_sequence(ball_batch, batch_first=True, padding_value=0)

    labels = torch.tensor(labels).float().unsqueeze(1)
    return team_batch, player_batch, ball_batch, labels

def partition_data_with_keys(df, group_keys):
    partitions = df.partition_by(group_keys)
    keys = [tuple(partition.select(group_keys).unique().to_numpy()[0]) for partition in partitions]
    partitions = [partition.drop(group_keys).to_numpy() for partition in partitions]
    return keys, partitions

# Extract data from train_dataset
def extract_data(dataset):
    team_stats_list = []
    player_stats_list = []
    ball_stats_list = []
    labels_list = []

    for idx in dataset.indices:
        team_input, player_input, ball_input, label = dataset.dataset[idx]
        team_stats_list.append(team_input.numpy())
        player_stats_list.append(player_input.numpy())
        ball_stats_list.append(ball_input.numpy())
        labels_list.append(label.item())
    
    return team_stats_list, player_stats_list, ball_stats_list, labels_list

def augment_match_data(balltoball, team_stats, player_stats, labels, window_sizes=[20,25,30,35,40,45]):
    """
    Augments data to simulate predictions at various match stages using sliding windows.
    
    Parameters:
    - balltoball: List of ball-by-ball data arrays.
    - team_stats: List of team statistics arrays.
    - player_stats: List of player statistics arrays.
    - labels: Array of match outcome labels.
    - window_sizes: List of window sizes (in overs).

    Returns:
    - Augmented balltoball, team_stats, player_stats, and labels.
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
    
    return augmented_balltoball, augmented_team_stats, augmented_player_stats, augmented_labels
