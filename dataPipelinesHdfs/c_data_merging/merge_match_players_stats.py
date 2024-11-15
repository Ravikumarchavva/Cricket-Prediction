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
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
import utils

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
        
        original_teams = team_a.unionByName(team_b).orderBy("Country", "Player")
        swapped_teams = team_b_swapped.unionByName(team_a_swapped).orderBy("Country")
        
        matchPlayers = original_teams.unionByName(swapped_teams).orderBy(["match_id", "flip", "Country"])
        matchPlayers = matchPlayers.select(["match_id", "flip", "Country", "Player", "Season"])
        
        # Step 6: Join with player statistics
        matchPlayersStats = matchPlayers.join(playerStats, on=['Player', 'Country', 'Season'], how='inner')
        matchPlayersStats = matchPlayersStats.sort("match_id", "flip")
        
        # Step 7: Filter matches with exactly 44 records
        match_id = matchPlayersStats.groupBy('match_id').count().filter(col('count') == 44).select('match_id')
        match_id_values = [row.match_id for row in match_id.collect()]
        matchPlayersStats = matchPlayersStats.filter(col('match_id').isin(match_id_values))
        
        # Step 8: Drop unnecessary columns
        matchPlayersStats = matchPlayersStats.drop('Country', 'Season', 'Player', 'Country')
        print(matchPlayersStats.show(5))
        print(matchPlayersStats.count())
        
        # Step 9: Save the merged data
        utils.spark_save_data(matchPlayersStats, config.MERGED_DATA_DIR, 'players_stats_flip.csv')
        logging.info('Match players stats data saved successfully.')
        
    except Exception as e:
        logging.error(f"Error in processing match players stats: {e}")
        raise

if __name__ == '__main__':
    process_match_players_stats()

