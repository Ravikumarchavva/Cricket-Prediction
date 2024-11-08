import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import logging
import torch.nn.utils.rnn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class CricketDataset(Dataset):
    def __init__(self, team_stats_list, player_stats_list, ball_by_ball_list, labels):
        self.team_stats_list = team_stats_list
        self.player_stats_list = player_stats_list
        self.ball_by_ball_list = ball_by_ball_list
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.team_stats_list[idx], self.player_stats_list[idx], self.ball_by_ball_list[idx], self.labels[idx]

# Load the data    
def load_data(train_path):
    team_stats = pl.read_csv(os.path.join(train_path, 'team12Stats.csv'))
    player_stats = pl.read_csv(os.path.join(train_path, 'playersStats.csv'))
    ball_by_ball = pl.read_csv(os.path.join(train_path, 'balltoball.csv'))
    return team_stats, player_stats, ball_by_ball

# Preprocess the data
def partition_data(df, group_keys):
    partitions = df.partition_by(group_keys)
    partition_list = [partition.drop(group_keys).to_numpy() for partition in partitions]
    return partition_list

# Create labels
def create_labels(ball_by_ball_list):
    labels = []
    ball_by_balls = []
    for partition in ball_by_ball_list:
        labels.append(partition[:,-1][0])
        ball_by_balls.append(partition[:,:-1])
    return ball_by_balls, labels

def custom_collate_fn(batch):
    team_stats_list, player_stats_list, ball_by_ball_list, labels_list = zip(*batch)
    # Convert lists to tensors
    team_stats = torch.stack([torch.tensor(ts) for ts in team_stats_list])
    player_stats = torch.stack([torch.tensor(ps) for ps in player_stats_list])
    # Pad 'ball_by_ball' sequences
    ball_by_ball_tensors = [torch.tensor(bbb) for bbb in ball_by_ball_list]
    ball_by_ball_padded = torch.nn.utils.rnn.pad_sequence(ball_by_ball_tensors, batch_first=True)
    # Create masks
    lengths = torch.tensor([len(seq) for seq in ball_by_ball_tensors])
    max_length = ball_by_ball_padded.size(1)
    mask = torch.arange(max_length).expand(len(lengths), max_length) < lengths.unsqueeze(1)
    labels = torch.tensor(labels_list)
    return team_stats, player_stats, ball_by_ball_padded, mask, labels

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(base_dir, 'data', '4_filteredData')
    logging.info('Loading data...')
    team_stats, player_stats, ball_by_ball = load_data(data_dir)
    logging.info('Partitioning data...')
    team_stats_list = partition_data(team_stats, ['match_id','flip'])
    player_stats_list = partition_data(player_stats, ['match_id','flip'])
    ball_by_ball_list = partition_data(ball_by_ball, ['match_id','flip'])
    logging.info('Data partitioned successfully!')
    logging.info('Creating labels...')
    ball_by_ball_list, labels = create_labels(ball_by_ball_list)
    logging.info('Labels created successfully!')
    logging.info('Creating dataset...')
    dataset = CricketDataset(team_stats_list, player_stats_list, ball_by_ball_list, labels)
    logging.info('Dataset created successfully!')   
    logging.info('Creating dataloader...')
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        prefetch_factor=2,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    logging.info('Dataloader created successfully!')
    logging.info('Iterating through the dataloader...')
    for i, data in enumerate(dataloader):
        team_stats, player_stats, ball_by_ball, mask, labels = data
        logging.info(f'Batch {i+1} - Team Stats: {team_stats.shape}, Player Stats: {player_stats.shape}, Ball By Ball: {ball_by_ball.shape}, Labels: {labels.shape}, mask: {mask.shape}')
        if i == 0:
            break
if __name__ == '__main__':
    main()