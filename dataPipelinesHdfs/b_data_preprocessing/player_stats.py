"""Module for processing and combining player statistics across different aspects of cricket."""

import os
import logging
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from utils import create_spark_session, load_data, spark_save_data, country_codes

from b_data_preprocessing.preprocessing import (
    preprocess_batting_data,
    preprocess_bowling_data,
    preprocess_fielding_data,
    map_country_codes
)

raw_data_dir = config.RAW_DATA_DIR
processed_data_dir = config.PROCESSED_DATA_DIR
country_codes = country_codes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_batting():
    """Process batting statistics and save to HDFS."""
    logging.info("Starting preprocess_batting task.")
    spark = create_spark_session("BattingStatsPreprocessing")

    try:
        batting_data = load_data(spark, raw_data_dir, 't20_batting_stats.csv')
        batting_data = preprocess_batting_data(batting_data)
        batting_data = map_country_codes(batting_data, country_codes)
        print(batting_data.show(5))
        spark_save_data(batting_data, processed_data_dir, 'batting_data.csv')
        logging.info("Batting data processing and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in preprocess_batting task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

def preprocess_bowling():
    """Process bowling statistics and save to HDFS."""
    logging.info("Starting preprocess_bowling task.")
    spark = create_spark_session("BowlingStatsPreprocessing")
    try:
        bowling_data = load_data(spark, raw_data_dir, 't20_bowling_stats.csv')
        bowling_data = preprocess_bowling_data(bowling_data)
        bowling_data = map_country_codes(bowling_data, country_codes)
        spark_save_data(bowling_data, processed_data_dir, 'bowling_data.csv')
        logging.info("Bowling data processing and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in preprocess_bowling task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

def preprocess_fielding():
    """Process fielding statistics and save to HDFS."""
    logging.info("Starting preprocess_fielding task.")
    spark = create_spark_session("FieldingStatsPreprocessing")

    try:
        fielding_data = load_data(spark, raw_data_dir, 't20_fielding_stats.csv')
        fielding_data = preprocess_fielding_data(fielding_data)
        fielding_data = map_country_codes(fielding_data, country_codes)
        spark_save_data(fielding_data, processed_data_dir, 'fielding_data.csv')
        logging.info("Fielding data processing and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in preprocess_fielding task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

def combine_data():
    """
    Combine batting, bowling, and fielding statistics into a single dataset.

    Merges processed statistics from different aspects of the game and joins with
    player information to create a comprehensive player statistics dataset.
    """
    logging.info("Starting combine_data task.")
    spark = create_spark_session("CombinePlayerStats")

    try:
        batting_data = load_data(spark, processed_data_dir, 'batting_data.csv')
        bowling_data = load_data(spark, processed_data_dir, 'bowling_data.csv')
        fielding_data = load_data(spark, processed_data_dir, 'fielding_data.csv')
        players_data = load_data(spark, processed_data_dir, 'players.csv')

        batting_data = batting_data.join(players_data, ['Player', 'Country', 'Season'], 'inner')
        bowling_data = bowling_data.join(players_data, ['Player', 'Country', 'Season'], 'inner')
        fielding_data = fielding_data.join(players_data, ['Player', 'Country', 'Season'], 'inner')

        bowling_data = bowling_data.drop('Mat','Player', 'Country', 'Inns')
        fielding_data = fielding_data.drop('Mat','Player', 'Country', 'Inns')   

        playerdata = batting_data.join(bowling_data, on=['player_id',"Season"], how='inner').join(fielding_data, on=['player_id',"Season"], how='inner')\
        .drop('Cumulative Mat','Cumulative Inns')

        player_data = playerdata.dropDuplicates(['player_id', 'Player', 'Country', 'Season'])
        print(player_data.show(5))
        
        player_data.show(5)

        spark_save_data(player_data, config.PROCESSED_DATA_DIR, 'player_stats.csv')
        logging.info("Data combining and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in combine_data task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

if __name__ == '__main__':
    # preprocess_batting()
    # preprocess_bowling()
    # preprocess_fielding()
    combine_data()