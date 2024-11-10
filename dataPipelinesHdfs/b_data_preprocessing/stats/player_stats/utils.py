
import os
import logging
from pyspark.sql import SparkSession

def create_spark_session():
    """Create and return a Spark session."""
    logging.info("Creating Spark session.")
    try:
        spark = SparkSession.builder \
            .appName("PlayerStats") \
            .master("spark://192.168.245.142:7077") \
            .config("spark.executor.memory", "2g") \
            .config("spark.executor.cores", "2") \
            .config("spark.cores.max", "4") \
            .getOrCreate()
        
        # Set Spark logging level to WARN
        spark.sparkContext.setLogLevel("WARN")
        logging.info("Spark session created successfully.")
        return spark
    except Exception as e:
        logging.error(f"Error creating Spark session: {e}")
        raise

def load_data(spark, data_dir, filename):
    """Load CSV data."""
    logging.info(f"Loading data from {filename}.")
    return spark.read.csv(
        os.path.join(data_dir, filename),
        header=True,
        inferSchema=True
    )

def save_data(df, hdfs_dir, filename):
    """Save DataFrame to HDFS as CSV."""
    logging.info(f"Saving data to {filename} in HDFS.")
    output_path = os.path.join(hdfs_dir, filename)
    try:
        # Write DataFrame to HDFS as CSV
        df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')
        logging.info(f"Data saved successfully to {filename} in HDFS.")
    except Exception as e:
        logging.error(f"An error occurred while saving data to {filename} in HDFS: {e}")