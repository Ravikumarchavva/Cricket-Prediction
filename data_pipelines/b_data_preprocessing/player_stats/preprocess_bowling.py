"""Preprocess bowling statistics and save to HDFS."""

import os
import sys
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import config
from utils import create_spark_session, load_data, spark_save_data, country_codes
from data_pipelines.b_data_preprocessing.player_stats.preprocessing import (
    preprocess_bowling_data,
    map_country_codes
)


def preprocess_bowling():
    """Process bowling statistics and save to HDFS."""
    logging.info("Starting preprocess_bowling task.")
    spark = create_spark_session("BowlingStatsPreprocessing")

    try:
        bowling_data = load_data(spark, config.RAW_DATA_DIR, 't20_bowling_stats.csv')

        # Data quality checks
        if bowling_data is None or bowling_data.count() == 0:
            logging.error("Bowling data is empty.")
            raise ValueError("Bowling data is empty.")
        required_columns = ["Player", "Season", "Mat", "Inns", "Overs", "Runs", "Wkts", "Econ"]
        missing_columns = [col for col in required_columns if col not in bowling_data.columns]
        if missing_columns:
            logging.error(f"Missing columns in bowling data: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        bowling_data = preprocess_bowling_data(bowling_data)
        bowling_data = map_country_codes(bowling_data, country_codes)

        # Add print statements to display columns and count
        print("Transformed bowling data columns:", bowling_data.columns)
        print("Transformed bowling data count:", bowling_data.count())

        spark_save_data(bowling_data, config.PROCESSED_DATA_DIR, 'bowling_data.csv')
        logging.info("Bowling data processing and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in preprocess_bowling task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")


if __name__ == '__main__':
    preprocess_bowling()