import os
import glob
import logging
from hdfs import InsecureClient
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when
import config

def preprocess_deliveries():
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime%s - %(levelname)s - %(message)s')

    try:
        # Initialize HDFS client
        client = InsecureClient(f'http://{config.HDFS_HOST}:{config.HDFS_HTTP_PORT}', user=config.HDFS_USER)
        hdfs_data_path = config.HDFS_BASE_DIR

        # Check the contents of the directory on HDFS
        logging.info(f'Checking contents of HDFS directory: {os.path.join(hdfs_data_path, "1_rawData", "t20s_csv2")}')
        dir_contents = client.list(os.path.join(hdfs_data_path, '1_rawData', 't20s_csv2'))

        # Find all CSV files in the specified directory
        info_files = [f for f in dir_contents if f.endswith('_info.csv')]
        delivery_files = [f for f in dir_contents if not f.endswith('_info.csv')]
        logging.info(f'Found {len(info_files)} info files and {len(delivery_files)} delivery files.')

        if len(delivery_files) == 0:
            logging.warning(f'No delivery files found in {os.path.join(hdfs_data_path, "1_rawData", "t20s_csv2")}. Please check the directory and file permissions.')

        # Initialize Spark session with HDFS configuration
        spark = SparkSession.builder \
            .appName(config.SPARK_APP_NAME) \
            .config("spark.hadoop.fs.defaultFS", config.HDFS_URI) \
            .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
            .config("spark.jars.packages", "org.slf4j:slf4j-jdk14:1.7.30") \
            .getOrCreate()

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
        delivery_paths = [os.path.join(hdfs_data_path, '1_rawData', 't20s_csv2', f) for f in delivery_files]
        deliveries_data = spark.read.csv(delivery_paths, header=True, schema=delivery_schema)

        # Fill null values
        deliveries_data = deliveries_data.fillna(0)

        # Convert specific columns to integer type
        deliveries_data = deliveries_data.withColumn('noballs', when(col('noballs').isNull(), '0').otherwise(col('noballs')).cast(IntegerType()))
        deliveries_data = deliveries_data.withColumn('penalty', when(col('penalty').isNull(), '0').otherwise(col('penalty')).cast(IntegerType()))

        columns = ['wicket_type', 'player_dismissed', 'other_wicket_type', 'other_player_dismissed']
        for column in columns:
            deliveries_data = deliveries_data.withColumn(column, when(col(column).isNull(), '0').otherwise('1').cast(IntegerType()))

        # Save Spark DataFrame to HDFS in CSV format
        try:
            deliveries_data.write.mode('overwrite').csv(f'{config.PROCESSED_DATA_DIR}/deliveries_csv', header=True)
            logging.info('Saved deliveries_csv to HDFS.')
        except Exception as e:
            logging.error(f'Error saving deliveries_csv to HDFS: {e}')
            raise

    except Exception as e:
        logging.critical(f'Critical error: {e}')
        raise

if __name__ == '__main__':
    preprocess_deliveries()

