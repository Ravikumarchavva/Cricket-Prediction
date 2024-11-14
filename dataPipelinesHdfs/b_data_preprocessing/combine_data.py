
"""Combine batting, bowling, and fielding statistics into a single dataset."""

import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import config

import config
from utils import create_spark_session, load_data, spark_save_data

def combine_data():
    """Combine batting, bowling, and fielding statistics into a single dataset."""
    logging.info("Starting combine_data task.")
    spark = create_spark_session("CombinePlayerStats")

    try:
        batting_data = load_data(spark, config.PROCESSED_DATA_DIR, 'batting_data.csv')
        bowling_data = load_data(spark, config.PROCESSED_DATA_DIR, 'bowling_data.csv')
        fielding_data = load_data(spark, config.PROCESSED_DATA_DIR, 'fielding_data.csv')
        players_data = load_data(spark, config.PROCESSED_DATA_DIR, 'match_players.csv')

        batting_data = batting_data.join(players_data, ['Player', 'Country', 'Season'], 'inner')
        bowling_data = bowling_data.join(players_data, ['Player', 'Country', 'Season'], 'inner')
        fielding_data = fielding_data.join(players_data, ['Player', 'Country', 'Season'], 'inner')

        bowling_data = bowling_data.drop('Mat', 'Player', 'Country', 'Inns')
        fielding_data = fielding_data.drop('Mat', 'Player', 'Country', 'Inns')

        player_data = batting_data.join(bowling_data, on=['player_id', 'Season'], how='inner') \
                                  .join(fielding_data, on=['player_id', 'Season'], how='inner') \
                                  .drop('Cumulative Mat', 'Cumulative Inns')

        player_data = player_data.dropDuplicates(['player_id', 'Player', 'Country', 'Season'])
        spark_save_data(player_data, config.PROCESSED_DATA_DIR, 'player_stats.csv')
        logging.info("Data combining and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in combine_data task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

if __name__ == '__main__':
    combine_data()