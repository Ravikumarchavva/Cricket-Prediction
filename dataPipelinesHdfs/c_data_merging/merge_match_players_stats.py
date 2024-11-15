"""Module for merging match player data with player statistics.

Processes player data, handles flipping for modeling, and integrates player statistics.
Uses configurations and utilities for session management and data operations.
"""

import os
import logging
from pyspark.sql import Window
from pyspark.sql.functions import col, lit, row_number

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
import utils

def process_match_players_stats():
    """Merge match players with their statistics."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    try:
        # Initialize Spark session using utils
        spark = utils.create_spark_session("MatchPlayersStats")

        # Step 1: Load data from HDFS using utils
        matchPlayers = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'match_players.csv')
        playerStats = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'player_stats.csv')
        
        # Add flip column
        matchPlayers = matchPlayers.withColumn("flip", lit(0))
        
        # Step 2: Define window specification
        window_spec = Window.partitionBy("match_id").orderBy("flip")
        
        # Step 3: Assign row numbers and split into teams
        matchPlayers = matchPlayers.withColumn("row_num", row_number().over(window_spec))
        
        team_a = matchPlayers.filter(col("row_num") <= 11).withColumn("flip", lit(0))
        team_b = matchPlayers.filter(col("row_num") > 11).withColumn("flip", lit(0))
        
        # Step 4: Create swapped teams with flip = 1
        team_b_swapped = team_a.withColumn("flip", lit(1))
        team_a_swapped = team_b.withColumn("flip", lit(1))
        
        original_teams = team_a.unionByName(team_b).orderBy("country", "player_id")
        swapped_teams = team_b_swapped.unionByName(team_a_swapped).orderBy("country")
        
        matchPlayers = original_teams.unionByName(swapped_teams).orderBy(["match_id", "flip", "country"])
        matchPlayers = matchPlayers.select(["match_id", "flip", "player_id", "country", "player", "season"])
        
        # Step 6: Join with player statistics
        matchPlayersStats = matchPlayers.join(playerStats, on=['player_id', 'season'], how='inner')
        matchPlayersStats = matchPlayersStats.sort("match_id", "flip")
        
        # Step 7: Filter matches with exactly 44 records
        match_id = matchPlayersStats.groupBy('match_id').count().filter(col('count') == 44).select('match_id')
        match_id_values = [row.match_id for row in match_id.collect()]
        matchPlayersStats = matchPlayersStats.filter(col('match_id').isin(match_id_values))
        
        # Step 8: Drop unnecessary columns
        matchPlayersStats = matchPlayersStats.drop('country', 'player', 'player_id', 'season', 'Player', 'Country')
        
        # Step 9: Save the merged data
        utils.spark_save_data(matchPlayersStats, config.MERGED_DATA_DIR, 'players_stats_flip.csv')
        logging.info('Match players stats data saved successfully.')
        
    except Exception as e:
        logging.error(f"Error in processing match players stats: {e}")
        raise

if __name__ == '__main__':
    process_match_players_stats()

