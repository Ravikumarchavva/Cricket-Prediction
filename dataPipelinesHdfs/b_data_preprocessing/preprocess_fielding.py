
"""Preprocess fielding statistics and save to HDFS."""

import os
import logging
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from utils import create_spark_session, load_data, spark_save_data, country_codes

from b_data_preprocessing.preprocessing import (
    preprocess_fielding_data,
    map_country_codes
)

def preprocess_fielding():
    """Process fielding statistics and save to HDFS."""
    logging.info("Starting preprocess_fielding task.")
    spark = create_spark_session("FieldingStatsPreprocessing")

    try:
        fielding_data = load_data(spark, config.RAW_DATA_DIR, 't20_fielding_stats.csv')
        fielding_data = preprocess_fielding_data(fielding_data)
        fielding_data = map_country_codes(fielding_data, country_codes)
        spark_save_data(fielding_data, config.PROCESSED_DATA_DIR, 'fielding_data.csv')
        logging.info("Fielding data processing and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in preprocess_fielding task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

if __name__ == '__main__':
    preprocess_fielding()