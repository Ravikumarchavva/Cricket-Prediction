import os
import logging
from pyspark.sql import SparkSession
from playerStats.main import create_spark_session, map_country_codes

def load_data(spark, data_dir, filename):
    """Load CSV data."""
    logging.info(f"Loading data from {filename}.")
    return spark.read.csv(
        os.path.join(data_dir, filename),
        header=True,
        inferSchema=True
    )

def load_players_data(spark, data_dir):
    """Load players data."""
    logging.info("Loading players data.")
    players_data = load_data(spark, data_dir, 'Players.csv')
    players_data = players_data.withColumnRenamed("player", "Player") \
                               .withColumnRenamed("country", "Country")
    return players_data
