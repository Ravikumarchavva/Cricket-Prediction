#!/usr/bin/env python
# coding: utf-8

# # Adding paths and files

# In[1]:


import os
import glob
import polars as pl

# Specify the directory where your CSV files are located
directory = r'D:\github\Cricket-Prediction\data\1_rawData\t20s_csv2'

# Use glob to find all CSV files in the specified directory
info_files = glob.glob(os.path.join(directory, '*_info.csv'))
all_files = glob.glob(os.path.join(directory,'*.csv'))
delivery_files = [file for file in all_files if '_info' not in file]

matches=[]
deliveries=[]
# Print the list of CSV files
for info_file in info_files:
    matches.append(info_file.split('\\')[-1])
for delivery in delivery_files:
    if '_info' not in delivery:
        deliveries.append(delivery.split('\\')[-1])


# In[2]:


import pandas as pd

df = pd.read_csv(info_files[0], header=None, names=['type', 'heading', 'subkey', 'players','player_id'], skipinitialspace=True).drop('type', axis=1)
df.head(10)


# In[3]:


df['subkey'][5]


# In[4]:


match_id = pd.to_numeric(info_files[0].split('\\')[-1].split('_')[0])
match_id


# In[5]:


# Filter dataframes based on the heading
players_df = df[df['heading'] == "player"].drop(['heading','player_id'], axis=1)
registry_df = df[df['heading'] == "registry"].drop('heading', axis=1)

# Join on the 'players' column with 'player_id' from the registry dataframe
merged_df = players_df.merge(registry_df[['players', 'player_id']], on='players', how='inner')

# Display the merged dataframe
merged_df.rename(columns={'players':'player','subkey':'country'}, inplace=True)
merged_df['match_id'] = match_id
merged_df


# In[6]:


dataframes = pd.DataFrame(columns=['country', 'player','player_id','season','match_id'])
injured_matches = []

for info_file in info_files:
    match_id = pd.to_numeric(info_file.split('\\')[-1].split('_')[0])
    try:
        df = pd.read_csv(info_file, header=None, names=['type', 'heading', 'subkey', 'players','player_id'], skipinitialspace=True).drop('type', axis=1)
        players_df = df[df['heading'] == "player"].drop(['heading','player_id'], axis=1)
        registry_df = df[df['heading'] == "registry"].drop('heading', axis=1)
        merged_df = players_df.merge(registry_df[['players', 'player_id']], on='players', how='inner')
        merged_df.rename(columns={'players':'player','subkey':'country'}, inplace=True)
        season = df['subkey'][5] 
        merged_df['match_id'] = match_id
        merged_df['season'] = season
        if(len(merged_df)!=22):
            raise Exception('Injured Match')
        dataframes = pd.concat([dataframes, merged_df])
    except:
        injured_matches.append(match_id)
print(injured_matches)


# In[7]:


dataframes


# In[8]:


len(dataframes)/22,len(injured_matches)


# In[9]:


dataframes.to_csv(os.path.join(directory,"../../2_processedData/Matchplayers.csv"),index=False)


# # Individual player's data

# In[10]:


players = pl.from_pandas(dataframes).drop('match_id').select('player','country','player_id').unique()
players


# In[11]:


players.write_csv(os.path.join(directory,'../../2_processedData/Players.csv'))

