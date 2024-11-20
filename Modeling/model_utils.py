import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence

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

def collate_fn_with_packing(batch):
    team_inputs = []
    player_inputs = []
    ball_inputs = []
    labels = []

    for team_input, player_input, ball_input, label in batch:
        team_inputs.append(team_input)
        player_inputs.append(player_input)
        ball_inputs.append(ball_input)
        labels.append(label)

    packed_ball_inputs = pack_sequence(ball_inputs, enforce_sorted=False)
    team_inputs = torch.stack(team_inputs)
    player_inputs = torch.stack(player_inputs)
    labels = torch.tensor(labels, dtype=torch.float32)

    return team_inputs, player_inputs, packed_ball_inputs, labels

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
