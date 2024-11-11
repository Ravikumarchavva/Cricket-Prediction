"""Module for preprocessing team statistics data for T20 cricket matches."""

import os
import sys
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import config
import utils

dfs_config = {
    'spark.executor.memory': '2g',
    'spark.executor.cores': '2',
    'spark.cores.max': '6',
    'spark.hadoop.fs.defaultFS': config.HDFS_URI,
    'spark.hadoop.fs.hdfs.impl': 'org.apache.hadoop.hdfs.DistributedFileSystem',
    'spark.hadoop.fs.hdfs.client.use.datanode.hostname': 'true',
    'spark.hadoop.dfs.client.use.datanode.hostname': 'true'
}

def preprocess_team_data():
    """
    Process team statistics data and calculate cumulative metrics.

    Reads raw team statistics, calculates various performance metrics including
    win/loss ratios and running averages, and saves processed data to HDFS.

    Returns:
        None
    """
    logging.info("Starting preprocess_team_data task.")
    try:
        # Create Spark session
        spark = utils.create_spark_session("TeamStatsPreprocessing", dfs_config)

        # Read team data
        team_data = utils.load_data(spark, config.RAW_DATA_DIR, 't20_team_stats.csv')

        from pyspark.sql.functions import col, when, round
        team_data = team_data.withColumn("W/L", round(when(col("Lost") == 0, col("Won")).otherwise(col("Won") / col("Lost")), 2))
        team_data = team_data.withColumn("AveRPW", when(col("Ave") == '-', 0).otherwise(col("Ave")).cast("float")).drop("Ave")
        team_data = team_data.withColumn("AveRPO", when(col("RPO") == '-', 0).otherwise(col("RPO")).cast("float")).drop("RPO", "LS")

        # Cumulative calculations
        from pyspark.sql import Window
        from pyspark.sql.functions import col, sum as spark_sum, when, row_number, round

        # Define the window specification for cumulative calculations
        window_spec = Window.partitionBy("Team").orderBy("Season").rowsBetween(Window.unboundedPreceding, -1)

        # Window for row number to identify the first row per player and country
        row_num_window = Window.partitionBy("Team").orderBy("Season")

        # perform cumulative calculations
        team_data = team_data.withColumn("row_num", row_number().over(row_num_window)) \
            .withColumn("Cumulative Won",
                        when(col("row_num") == 1, 0)
                        .otherwise(spark_sum("Won").over(window_spec))) \
            .withColumn("Cumulative Lost",
                        when(col("row_num") == 1, 0)  # Set 0 for the first row (before any match)
                        .otherwise(spark_sum("Lost").over(window_spec))) \
            .withColumn("Cumulative Tied", 
                        when(col("row_num") == 1, 0)  # Set 0 for the first row (before any match)
                        .otherwise(spark_sum("Tied").over(window_spec))) \
            .withColumn("Cumulative NR", 
                        when(col("row_num") == 1, 0)
                        .otherwise(spark_sum("NR").over(window_spec))) \
            .withColumn("Cumulative W/L", 
                        when(col("row_num") == 1, 0)
                        .otherwise(
                            round(
                                when(spark_sum("Lost").over(window_spec) != 0, 
                                     spark_sum(("Won")).over(window_spec) / spark_sum("Lost").over(window_spec))
                                .otherwise(0), 2)
                        )
            ) \
            .withColumn("Cumulative AveRPW", 
                        when(col("row_num") == 1, 0)
                        .otherwise(
                            round(
                                when(spark_sum("Won").over(window_spec) != 0, 
                                     spark_sum(col("AveRPW")*col("Mat")).over(window_spec) / spark_sum("Mat").over(window_spec))
                                .otherwise(0), 2)
                        )
            ) \
            .withColumn("Cumulative AveRPO", 
                        when(col("row_num") == 1, 0)
                        .otherwise(
                            round(
                                when(spark_sum("Lost").over(window_spec) != 0, 
                                     spark_sum(col("AveRPO")*col("Mat")).over(window_spec) / spark_sum("Mat").over(window_spec))
                                .otherwise(0), 2)
                        )
            ) \
            .drop("row_num")  # Drop the temporary row number column

        team_data = team_data.select("Team", "Season", "Cumulative Won", "Cumulative Lost", "Cumulative Tied", "Cumulative W/L", "Cumulative AveRPW", "Cumulative AveRPO")

        # Write processed data
        utils.save_data(team_data, config.PROCESSED_DATA_DIR, 'team_stats.csv')
        logging.info("Team data processing and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in preprocess_team_data task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

if __name__ == "__main__":
    preprocess_team_data()