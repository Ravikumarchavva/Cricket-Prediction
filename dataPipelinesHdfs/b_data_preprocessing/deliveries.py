"""Module for preprocessing ball-by-ball delivery data from T20 cricket matches.

Handles the processing of delivery-level data including runs, wickets, and other
ball-by-ball events in cricket matches.
"""

import os
import logging
from pyspark.sql.types import (
    StructType, StructField, IntegerType,
    StringType, FloatType
)
from pyspark.sql.functions import col, when

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
import utils


def preprocess_deliveries():
    """
    Process ball-by-ball delivery data from cricket matches.

    Reads raw delivery data files, processes them to extract ball-by-ball
    information including runs, wickets, and other events, and saves the
    processed data to HDFS.

    Returns:
        None
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Initialize HDFS client
        client = utils.get_hdfs_client()

        # Check the contents of the directory on HDFS
        logging.info(f'Checking contents of HDFS directory: {os.path.join(config.RAW_DATA_DIR, "t20s_csv2")}')
        dir_contents = utils.hdfs_list(client, os.path.join(config.RAW_DATA_DIR, 't20s_csv2'))

        # Find all CSV files in the specified directory
        info_files = [f for f in dir_contents if f.endswith('_info.csv')]
        delivery_files = [f for f in dir_contents if not f.endswith('_info.csv')]
        logging.info(f'Found {len(info_files)} info files and {len(delivery_files)} delivery files.')

        if len(delivery_files) == 0:
            logging.warning(f'No delivery files found in {os.path.join(config.RAW_DATA_DIR, "t20s_csv2")}. Please check the directory and file permissions.')

        # Initialize Spark session with HDFS configuration
        spark = utils.create_spark_session()

        # Define the schema for the deliveries data
        delivery_schema = StructType([
            StructField('match_id', IntegerType(), True),
            StructField('season', StringType(), True),
            StructField('start_date', StringType(), True),
            StructField('venue', StringType(), True),
            StructField('innings', IntegerType(), True),
            StructField('ball', FloatType(), True),
            StructField('batting_team', StringType(), True),
            StructField('bowling_team', StringType(), True),
            StructField('striker', StringType(), True),
            StructField('non_striker', StringType(), True),
            StructField('bowler', StringType(), True),
            StructField('runs_off_bat', IntegerType(), True),
            StructField('extras', IntegerType(), True),
            StructField('wides', IntegerType(), True),
            StructField('noballs', StringType(), True),
            StructField('byes', IntegerType(), True),
            StructField('legbyes', IntegerType(), True),
            StructField('penalty', StringType(), True),
            StructField('wicket_type', StringType(), True),
            StructField('player_dismissed', StringType(), True),
            StructField('other_wicket_type', StringType(), True),
            StructField('other_player_dismissed', StringType(), True)
        ])
        # Read the delivery files into a Spark DataFrame
        delivery_paths = [os.path.join(config.RAW_DATA_DIR, 't20s_csv2', f) for f in delivery_files]
        deliveries_data = spark.read.csv(delivery_paths, header=True, schema=delivery_schema)
        # Fill null values
        deliveries_data = deliveries_data.fillna(0)

        # Convert specific columns to integer type
        deliveries_data = deliveries_data.withColumn(
            'noballs',
            when(col('noballs').isNull(), '0').otherwise(col('noballs')).cast(IntegerType())
        )
        deliveries_data = deliveries_data.withColumn(
            'penalty',
            when(col('penalty').isNull(), '0').otherwise(col('penalty')).cast(IntegerType())
        )

        columns = ['wicket_type', 'player_dismissed', 'other_wicket_type', 'other_player_dismissed']
        for column in columns:
            deliveries_data = deliveries_data.withColumn(column, when(col(column).isNull(), '0').otherwise('1').cast(IntegerType()))

        # Save Spark DataFrame to HDFS in CSV format
        try:
            utils.spark_save_data(deliveries_data, config.PROCESSED_DATA_DIR, 'deliveries.csv')
            logging.info('Saved deliveries.csv to HDFS.')
        except Exception as e:
            logging.error(f'Error saving deliveries.csv to HDFS: {e}')
            raise

    except Exception as e:
        logging.critical(f'Critical error: {e}')
        raise


if __name__ == '__main__':
    preprocess_deliveries()

