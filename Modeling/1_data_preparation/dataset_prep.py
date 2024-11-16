import os

import polars as pl
# import data
def load_data():
    balltoball = pl.read_csv(os.path.join(os.path.join( '..',"data", "filtered_data" , "balltoball.csv")))
    team_stats = pl.read_csv(os.path.join(os.path.join( '..',"data", "filtered_data" , "team12_stats.csv")))
    players_stats = pl.read_csv(os.path.join(os.path.join( '..',"data", "filtered_data" , "players_stats.csv")))
    return balltoball, team_stats, players_stats
balltoball,team_stats,players_stats = load_data()


def partition_data(df, group_keys):
    partitions = df.partition_by(group_keys)
    partition_list = [partition.drop(group_keys).to_numpy() for partition in partitions]
    sequence_lengths = [len(partition) for partition in partitions]
    return partition_list, sequence_lengths

balltoball_partitions, balltoball_lengths = partition_data(balltoball, ["match_id","flip"])
team_stats_partitions, team_stats_lengths = partition_data(team_stats, ["match_id","flip"])
players_stats_partitions, players_stats_lengths = partition_data(players_stats, ["match_id","flip"])

import numpy as np
labels = np.array([])
for i in range(0, len(balltoball_partitions)):
    labels = np.append(labels, balltoball_partitions[i][0,-1])
    balltoball_partitions[i] = balltoball_partitions[i][:,:-1]

#print(data)

print(balltoball_partitions[0].shape) 
print(balltoball_partitions[0])