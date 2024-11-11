#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("matchPlayersStats").getOrCreate()

directory = r'D:\github\Cricket-Prediction\data\2_processedData'  # for local
# directory = '/app/dataInHandNow/afterpreprocessed'  # for docker

matches = spark.read.csv(os.path.join(directory, 'matches.csv'), header=True, inferSchema=True)
matchPlayers = spark.read.csv(os.path.join(directory, 'Matchplayers.csv'), header=True, inferSchema=True).sort('match_id')
playerStats = spark.read.csv(os.path.join(directory, 'playerStats.csv'), header=True, inferSchema=True)
playerStats.show(5)


# In[13]:


from pyspark.sql import functions as F

matchPlayers = matchPlayers.withColumn("flip", F.lit(0))
matchPlayers.show(5)


# In[14]:


from pyspark.sql import Window
from pyspark.sql.functions import col, lit, row_number

# Step 1: Create a window to assign row numbers within each match_id
window_spec = Window.partitionBy("match_id").orderBy("flip")

# Step 2: Assign row numbers to divide into two teams within each match_id
matchPlayers = matchPlayers.withColumn("row_num", row_number().over(window_spec))

# Step 3: Split data into Team A and Team B based on row number
team_a = matchPlayers.filter(col("row_num") <= 11).withColumn("flip", lit(0))  # Original Team A
team_b = matchPlayers.filter(col("row_num") > 11).withColumn("flip", lit(0))  # Original Team B

# Step 4: Create swapped teams with opposite order
team_b_swapped = team_a.withColumn("flip", lit(1))  # Team B followed by Team A (swapped)
team_a_swapped = team_b.withColumn("flip", lit(1))

# Step 5: Concatenate the original and swapped dataframes
original_teams = team_a.unionByName(team_b).orderBy("country", "player_id")  # Order by country and player_id in the original order
swapped_teams = team_b_swapped.unionByName(team_a_swapped).orderBy("country")  # Order by country and player_id in the swapped order

# Step 6: Combine original and swapped teams, ordering by match_id, flip, and player_id
matchPlayers = original_teams.unionByName(swapped_teams).orderBy(["match_id", "flip", "country"])

# Select the desired columns and display the result
matchPlayers = matchPlayers.select(["match_id", "flip", "player_id", "country", "player", "season"])
matchPlayers.show(44)


# In[15]:


playerStats.show(5)


# In[16]:


# Include row_num in the join
matchPlayersStats = matchPlayers.join(playerStats, on=['player_id','season'], how='inner')
matchPlayersStats = matchPlayersStats.sort("match_id", "flip")

# Display the result starting from the 45th row
matchPlayersStats.offset(44).show(44)


# In[17]:


match_id = matchPlayersStats.groupBy('match_id').count().filter(col('count') == 44).select('match_id')
match_id_list = match_id.collect()
len(match_id_list)


# In[18]:


# Extract match_id values from the collected rows
match_id_values = [row.match_id for row in match_id_list]

# Filter matchPlayersStats using the extracted match_id values
matchPlayersStats = matchPlayersStats.filter(col('match_id').isin(match_id_values))
matchPlayersStats.show(5)


# In[19]:


matchPlayersStats = matchPlayersStats.drop('country','player','player_id','season','Player','Country')
matchPlayersStats.show()


# In[20]:


num_rows = matchPlayersStats.count()
num_cols = len(matchPlayersStats.columns)
(num_rows, num_cols)


# In[21]:


directory = r'D:\github\Cricket-Prediction\data\3_aftermerging'  # for local
matchPlayersStats.toPandas().to_csv(os.path.join(directory, 'playersStatsflip.csv'))

