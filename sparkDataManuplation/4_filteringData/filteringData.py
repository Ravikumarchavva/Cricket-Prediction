#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append(os.path.join(os.getcwd(),"..",".."))
import utils, config

spark = utils.create_spark_session('filtering_data')

directory = config.MERGED_DATA_DIR

team12Stats = spark.read.csv(os.path.join(directory, 'team_stats.csv'), header=True, inferSchema=True)
balltoball = spark.read.csv(os.path.join(directory, 'ball_by_ball.csv'), header=True, inferSchema=True)
playerStats = spark.read.csv(os.path.join(directory, 'player_stats.csv'), header=True, inferSchema=True)


# In[2]:


team12Stats.select('match_id').distinct().count(), balltoball.select('match_id').distinct().count(), playerStats.select('match_id').distinct().count()


# In[3]:


# Extract match_id columns from each dataset
team12_match_ids = team12Stats.select('match_id').distinct()
balltoball_match_ids = balltoball.select('match_id').distinct()
player_match_ids = playerStats.select('match_id').distinct()

# Find intersection of match_id across all three datasets
common_match_ids = team12_match_ids.intersect(balltoball_match_ids).intersect(player_match_ids)
common_match_ids.collect()

# convert to list
common_match_ids_list = [row['match_id'] for row in common_match_ids.collect()]

print(common_match_ids_list),len(common_match_ids_list)


# In[4]:


# filter team12Stats

filtered_team12Stats = team12Stats.filter(team12Stats.match_id.isin(common_match_ids_list)).drop("_c0")

# filter balltoball

filtered_balltoball = balltoball.filter(balltoball.match_id.isin(common_match_ids_list)).drop("_c0")

# filter playersStats

filtered_playerStats = playerStats.filter(playerStats.match_id.isin(common_match_ids_list)).drop("_c0")


# In[5]:


print((filtered_playerStats.count(), len(filtered_playerStats.columns)), 
(filtered_team12Stats.count(), len(filtered_team12Stats.columns)), 
(filtered_balltoball.select('match_id').distinct().count(), len(filtered_balltoball.columns)))


# In[6]:


filtered_team12Stats.show(2)
filtered_balltoball.show(2)
filtered_playerStats.show(2)


# In[ ]:


filtered_balltoball = filtered_balltoball.select('match_id','innings','ball',"runs","wickets",'curr_score','curr_wickets','run_rate','required_run_rate','target',"won")

utils.spark_save_data(filtered_team12Stats, config.FILTERED_DATA_DIR, 'team12_stats.csv')
utils.spark_save_data(filtered_balltoball, config.FILTERED_DATA_DIR, 'ball_to_ball.csv')
utils.spark_save_data(filtered_playerStats, config.FILTERED_DATA_DIR, 'players_stats.csv')
spark.stop()


# In[ ]:




