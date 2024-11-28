"""Module for merging match player data with player statistics.

Processes player data, handles flipping for modeling, and integrates player statistics.
Uses configurations and utilities for session management and data operations.
"""

import os
import logging
from pyspark.sql import Window
from pyspark.sql.functions import col, lit, row_number
import pyspark.sql.functions as F

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from configs import spark_config as config
from utils import spark_utils as utils

def process_match_players_stats():
    """Merge match players with their statistics."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    try:
        # Initialize Spark session using utils
        spark = utils.create_spark_session("MatchPlayersStats",{
            "spark.executor.memory": "3g",
        })

        # Step 1: Load data from HDFS using utils
        matchPlayers = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'match_players.csv')
        playerStats = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'player_stats.csv')
        
        # Data quality checks and logging
        matchPlayers_rows, matchPlayers_cols = matchPlayers.count(), len(matchPlayers.columns)
        playerStats_rows, playerStats_cols = playerStats.count(), len(playerStats.columns)
        logging.info(f'MatchPlayers data: {matchPlayers_rows} rows, {matchPlayers_cols} columns')
        logging.info(f'PlayerStats data: {playerStats_rows} rows, {playerStats_cols} columns')
        
        # Check for nulls in critical columns
        matchPlayers_nulls = matchPlayers.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in matchPlayers.columns])
        playerStats_nulls = playerStats.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in playerStats.columns])
        logging.info('Null values in matchPlayers data:')
        matchPlayers_nulls.show()
        logging.info('Null values in playerStats data:')
        playerStats_nulls.show()
        
        # After loading matchPlayers, rename columns to match playerStats
        matchPlayers = matchPlayers.withColumnRenamed("country", "Country") \
                                   .withColumnRenamed("player", "Player") \
                                   .withColumnRenamed("season", "Season")
        
        # Join with player statistics
        matchPlayersStats = matchPlayers.join(playerStats, on=['Player', 'Country', 'Season'], how='inner')
        matchPlayersStats = matchPlayersStats.sort("match_id")

        # Filter matches with exactly 22 players before processing flips
        match_id = matchPlayers.groupBy('match_id').count().filter(col('count') == 22).select('match_id')
        match_id_values = [row.match_id for row in match_id.collect()]
        matchPlayers = matchPlayers.filter(col('match_id').isin(match_id_values))
        
        # Drop unnecessary columns
        matchPlayersStats = matchPlayersStats.drop('Country', 'Player', 'player_id', 'Season')
        print(matchPlayersStats.show(5))
        print(matchPlayersStats.count())
        
        # Save the merged data
        utils.spark_save_data(matchPlayersStats, config.MERGED_DATA_DIR, 'players_stats.csv')
        logging.info('Match players stats data saved successfully.')
        
    except Exception as e:
        logging.error(f"Error in processing match players stats: {e}")
        raise

if __name__ == '__main__':
    process_match_players_stats()

