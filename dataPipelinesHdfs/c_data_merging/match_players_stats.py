"""Module for merging match player data with player statistics.

Processes player data, handles flipping for modeling, and integrates player statistics.
Uses configurations and utilities for session management and data operations.
"""

import os
import logging
from pyspark.sql import Window
from pyspark.sql.functions import col, lit, row_number

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
import utils

def process_match_players_stats():
    """Merge match players with their statistics."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Initialize Spark session using utils
        spark = utils.create_spark_session()

        # Load data from HDFS using utils
        matches = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'matches.csv')
        matchPlayers = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'Matchplayers.csv').sort('match_id')
        playerStats = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'playerStats.csv')
        
        # Add flip column
        matchPlayers = matchPlayers.withColumn("flip", lit(0))
        
        # Create flipped teams
        window_spec = Window.partitionBy("match_id").orderBy("flip")
        matchPlayers = matchPlayers.withColumn("row_num", row_number().over(window_spec))
        
        team_a = matchPlayers.filter(col("row_num") <= 11).withColumn("flip", lit(0))
        team_b = matchPlayers.filter(col("row_num") > 11).withColumn("flip", lit(0))
        
        team_b_swapped = team_a.withColumn("flip", lit(1))
        team_a_swapped = team_b.withColumn("flip", lit(1))
        
        original_teams = team_a.unionByName(team_b).orderBy("country", "player_id")
        swapped_teams = team_b_swapped.unionByName(team_a_swapped).orderBy("country")
        
        matchPlayers = original_teams.unionByName(swapped_teams).orderBy(["match_id", "flip", "country"])
        matchPlayers = matchPlayers.select(["match_id", "flip", "player_id", "country", "player", "season"])
        
        # Join with player statistics
        matchPlayersStats = matchPlayers.join(playerStats, on=['player_id', 'season'], how='inner')
        matchPlayersStats = matchPlayersStats.sort("match_id", "flip")
        
        # Filter matches with complete data
        match_id = matchPlayersStats.groupBy('match_id').count().filter(col('count') == 44).select('match_id')
        match_id_values = [row.match_id for row in match_id.collect()]
        matchPlayersStats = matchPlayersStats.filter(col('match_id').isin(match_id_values))
        
        # Select relevant columns
        matchPlayersStats = matchPlayersStats.drop('country', 'player', 'player_id', 'season', 'Player', 'Country')
        
        # Save the combined data to HDFS using utils
        utils.save_data(matchPlayersStats, config.MERGED_DATA_DIR, 'players_stats_flip.csv')
        logging.info('Match players stats data saved successfully.')
        
    except Exception as e:
        logging.error(f"Error in processing match players stats: {e}")
        raise

if __name__ == '__main__':
    process_match_players_stats()

