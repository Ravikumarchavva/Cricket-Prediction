"""Preprocess fielding statistics and save to HDFS."""

import os
import sys
import logging
from pyspark.sql import SparkSession, DataFrame

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
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

        # Data quality checks
        if fielding_data is None or fielding_data.rdd.isEmpty():
            logging.error("Fielding data is empty.")
            raise ValueError("Fielding data is empty.")
        required_columns = ["Player", "Mat", "Inns", "Dis", "Ct", "St", "D/I", "Season"]
        missing_columns = [col for col in required_columns if col not in fielding_data.columns]
        if missing_columns:
            logging.error(f"Missing columns in fielding data: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        fielding_data = preprocess_fielding_data(fielding_data)
        fielding_data = map_country_codes(fielding_data, country_codes)
        print("Fielding Data",fielding_data.count())
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