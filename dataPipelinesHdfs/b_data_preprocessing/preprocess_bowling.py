
"""Preprocess bowling statistics and save to HDFS."""

import os
import logging
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import config
from utils import create_spark_session, load_data, spark_save_data, country_codes

from b_data_preprocessing.preprocessing import (
    preprocess_bowling_data,
    map_country_codes
)

def preprocess_bowling():
    """Process bowling statistics and save to HDFS."""
    logging.info("Starting preprocess_bowling task.")
    spark = create_spark_session("BowlingStatsPreprocessing")

    try:
        bowling_data = load_data(spark, config.RAW_DATA_DIR, 't20_bowling_stats.csv')
        bowling_data = preprocess_bowling_data(bowling_data)
        bowling_data = map_country_codes(bowling_data, country_codes)
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