"""Module for merging matches and deliveries data into a single dataset.

Uses Spark operations to preprocess and merge data, utilizing configurations and utilities 
for session management and HDFS operations.
"""

import os
import logging
from pyspark.sql import Window
from pyspark.sql.functions import (
    coalesce, col, lit, sum as F_sum, when, last, max as F_max
)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
import utils

def merge_data():
    """Process and merge matches and deliveries data."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Initialize Spark session using utils
        spark = utils.create_spark_session("MergeBallByBall")

        # Load preprocessed data from HDFS using utils
        matches = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'matches.csv')
        deliveries = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'deliveries.csv')
        
        # Data preprocessing steps
        deliveries = deliveries.drop('season', 'start_date', 'venue', 'striker', 'non_striker', 'bowler')
        
        # Calculate "runs" and "wickets"
        deliveries = deliveries.withColumn(
            "runs",
            coalesce(col("runs_off_bat"), lit(0)) +
            coalesce(col("extras"), lit(0)) +
            coalesce(col("wides"), lit(0)) +
            coalesce(col("noballs"), lit(0)) +
            coalesce(col("byes"), lit(0)) +
            coalesce(col("legbyes"), lit(0)) +
            coalesce(col("penalty"), lit(0))
        ).drop("runs_off_bat", "extras", "wides", "noballs", "byes", "legbyes", "penalty")
        
        deliveries = deliveries.withColumn(
            "wickets",
            (coalesce(col("player_dismissed").cast("int"), lit(0)) +
             coalesce(col("other_player_dismissed").cast("int"), lit(0)))
        ).drop("wicket_type", "player_dismissed", "other_wicket_type", "other_player_dismissed")
        
        # Compute cumulative sums
        window_spec = Window.partitionBy("match_id", "innings").orderBy("ball")
        deliveries = deliveries.withColumn("curr_score", F_sum("runs").over(window_spec))
        deliveries = deliveries.withColumn("curr_wickets", F_sum("wickets").over(window_spec))
        deliveries = deliveries.drop("runs", "wickets")
        
        # Join deliveries with matches data
        data = deliveries.join(matches, on='match_id').drop('season', 'venue', 'gender')
        
        # Create flipped dataframes for modeling
        data1 = data.withColumn("flip", lit(0))
        data2 = data1.withColumnRenamed("team1", "team_temp") \
                     .withColumnRenamed("team2", "team1") \
                     .withColumnRenamed("team_temp", "team2") \
                     .withColumn("flip", lit(1))
        
        data_combined = data1.unionByName(data2).sort('match_id') \
            .withColumn("won", when(col('winner') == col('team1'), 1).otherwise(0))
        
        data = data_combined.select('match_id', 'flip', 'innings', 'ball', 'curr_score', 'curr_wickets', 'won')
        data = data.sort('match_id', 'flip', 'innings', 'ball')
        
        # Calculate target scores
        window_spec_ffill = Window.partitionBy("match_id").orderBy("flip", "innings", "ball").rowsBetween(Window.unboundedPreceding, 0)
        data = data.withColumn(
            "target",
            when(
                (col("innings") == 1) & (col("curr_score") == F_max("curr_score").over(window_spec)),
                col("curr_score")
            ).otherwise(lit(None))
        )
        data = data.withColumn("target", last("target", ignorenulls=True).over(window_spec_ffill))
        data = data.withColumn("target", when(col("innings") == 1, 0).otherwise(col("target")))
        
        # Save the merged data to HDFS using utils
        utils.spark_save_data(data, config.MERGED_DATA_DIR, 'ball_by_ball_flip.csv')
        logging.info('Merged data saved successfully.')
        
    except Exception as e:
        logging.error(f"Error in merging data: {e}")
        raise

if __name__ == '__main__':
    merge_data()




