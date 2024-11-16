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
    
from torch.nn.utils.rnn import pack_sequence

def collate_fn_with_padding(batch):
    team_inputs = []
    player_inputs = []
    ball_inputs = []
    labels = []
    ball_lengths = []

    for team_input, player_input, ball_input, label in batch:
        team_inputs.append(team_input)
        player_inputs.append(player_input)
        ball_inputs.append(ball_input)
        labels.append(label)
        ball_lengths.append(ball_input.shape[0])

    max_seq_len = max(ball_lengths)
    feature_dim = ball_inputs[0].shape[1]

    # Pad ball inputs and create a mask
    padded_ball_inputs = torch.zeros(len(ball_inputs), max_seq_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(ball_inputs), max_seq_len, dtype=torch.bool)

    for i, ball_input in enumerate(ball_inputs):
        seq_len = ball_input.shape[0]
        padded_ball_inputs[i, :seq_len, :] = ball_input
        mask[i, :seq_len] = True  # True for valid entries

    team_inputs = torch.stack(team_inputs)
    player_inputs = torch.stack(player_inputs)
    labels = torch.tensor(labels, dtype=torch.float32)

    return team_inputs, player_inputs, padded_ball_inputs, labels, mask