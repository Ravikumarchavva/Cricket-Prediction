"""Module for merging matches and deliveries data into a single dataset.

Uses Spark operations to preprocess and merge data, utilizing configurations and utilities 
for session management and HDFS operations.
"""

import os
import logging
from pyspark.sql import Window
from pyspark.sql.functions import (
    coalesce, col, lit, sum as F_sum, when, last, max as F_max, count
)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
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
        
        # Drop unwanted columns from matches
        matches = matches.drop('date', 'city', 'toss_winner', 'toss_decision')
        # matches.show(5)
        
        # Drop unwanted columns from deliveries
        deliveries = deliveries.drop('season', 'start_date', 'venue', 'striker', 'non_striker', 'bowler')
        # deliveries.show(5)
        
        from pyspark.sql import Window
        from pyspark.sql.functions import coalesce, col, lit, sum as F_sum, when, last, max as F_max
        
        # Calculate "runs" as the row-wise sum of specified columns
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
        
        # Calculate "wickets" as the row-wise sum of dismissals, handling null values
        deliveries = deliveries.withColumn(
            "wickets",
            (coalesce(col("player_dismissed").cast("int"), lit(0)) +
             coalesce(col("other_player_dismissed").cast("int"), lit(0)))
        ).drop("wicket_type", "player_dismissed", "other_wicket_type", "other_player_dismissed")
        # deliveries.show(5)
    
        # Define the window specifications for cumulative sums partitioned by "match_id" and "innings"
        window_spec = Window.partitionBy("match_id", "innings").orderBy("ball")
    
        # Calculate cumulative sum for "runs" as "curr_score"
        deliveries = deliveries.withColumn(
            "curr_score",
            F_sum("runs").over(window_spec)
        )
    
        # Calculate cumulative sum for "wickets" as "curr_wickets"
        deliveries = deliveries.withColumn(
            "curr_wickets",
            F_sum("wickets").over(window_spec)
        )
    
        # Join deliveries with matches data
        data = deliveries.join(matches, on='match_id').drop('season', 'venue', 'gender', 'team1', 'team2')
        # data.sort('match_id').show(10)
    
        # Add "won" column
        data = data.withColumn(
            "won",
            when(data["batting_team"] == data["winner"], 1).otherwise(0)
        ).drop("batting_team", "bowling_team", "winner")
        # data.sort("match_id").show(10)
    
        window_spec = Window.partitionBy("match_id").orderBy("innings", "ball")
        window_spec_ffill = Window.partitionBy("match_id").orderBy("innings", "ball").rowsBetween(Window.unboundedPreceding, 0)
    
        # Calculate "target" score
        data = data.withColumn(
            "target",
            when(
                (col("innings") == 1) & (col("curr_score") == F_max("curr_score").over(window_spec)),
                col("curr_score")
            ).otherwise(lit(None))
        )
        data = data.withColumn("overs", col("ball").cast("int"))
        data = data.withColumn("run_rate",
                               when(col("overs") != 0, col("curr_score") / col("overs"))
                               .otherwise(0).cast("float"))
    
        # Forward fill the "target" column
        data = data.withColumn("target", last("target", ignorenulls=True).over(window_spec_ffill))
        data = data.withColumn("target", when(col("innings") == 1, 0).otherwise(col("target")))
        data = data.orderBy(col("match_id"), col("innings"), col("ball"))
    
        # Calculate "required_run_rate"
        data = data.withColumn("required_run_rate",
                               when(col("innings") == 1, 0)
                               .otherwise((col("target") - col("curr_score")) / (20 - col("overs"))).cast("float"))
    
        data = data.select(
            "match_id", "innings", "ball", "runs", "wickets", "curr_score", "curr_wickets",
            "overs", "run_rate", "required_run_rate", "target", "won"
        )
        # data.count()
        # data.show(240)
    
        # Save the merged data to HDFS using utils
        utils.spark_save_data(data, config.MERGED_DATA_DIR, "ball_by_ball.csv")
        logging.info('Merged data saved successfully.')
        
    except Exception as e:
        logging.error(f"Error in merging data: {e}")
        raise

if __name__ == '__main__':
    merge_data()




