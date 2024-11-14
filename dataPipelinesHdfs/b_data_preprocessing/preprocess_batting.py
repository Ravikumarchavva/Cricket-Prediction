
"""Preprocess batting statistics and save to HDFS."""

import os
import logging
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import config
from utils import create_spark_session, load_data, spark_save_data, country_codes

from b_data_preprocessing.preprocessing import (
    preprocess_batting_data,
    map_country_codes
)

def preprocess_batting():
    """Process batting statistics and save to HDFS."""
    logging.info("Starting preprocess_batting task.")
    spark = create_spark_session("BattingStatsPreprocessing")

    try:
        batting_data = load_data(spark, config.RAW_DATA_DIR, 't20_batting_stats.csv')
        batting_data = preprocess_batting_data(batting_data)
        batting_data = map_country_codes(batting_data, country_codes)
        spark_save_data(batting_data, config.PROCESSED_DATA_DIR, 'batting_data.csv')
        logging.info("Batting data processing and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in preprocess_batting task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

if __name__ == '__main__':
    preprocess_batting()