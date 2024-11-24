"""Module for processing team statistics data."""

import os
import sys
import logging
from pyspark.sql.functions import col, when, round, sum as spark_sum, row_number
from pyspark.sql import Window

# Adjust sys.path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))

import config
import utils


def preprocess_team_data():
    """Process and transform team statistics data."""
    logging.info("Starting preprocess_team_data task.")

    try:
        spark = utils.create_spark_session("TeamStatsPreprocessing", {
            "spark.executor.memory": "512m",
        })

        # Read team data
        team_data = utils.load_data(spark, config.RAW_DATA_DIR, 't20_team_stats.csv')

        # Data quality checks
        if team_data is None or team_data.count() == 0:
            logging.error("Team data is empty.")
            raise ValueError("Team data is empty.")
        required_columns = ["Team", "Season", "Mat", "Won", "Lost", "Tied", "NR", "Ave", "RPO"]
        missing_columns = [col for col in required_columns if col not in team_data.columns]
        if missing_columns:
            logging.error(f"Missing columns in team data: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        team_data = team_data.withColumn(
            "W/L",
            round(
                when(col("Lost") == 0, col("Won")).otherwise(col("Won") / col("Lost")), 2
            )
        )
        team_data = team_data.withColumn(
            "AveRPW", when(col("Ave") == '-', 0).otherwise(col("Ave")).cast("float")).drop("Ave")
        team_data = team_data.withColumn(
            "AveRPO", when(col("RPO") == '-', 0).otherwise(col("RPO")).cast("float")).drop("RPO", "LS")

        # Cumulative calculations

        # Define the window specification for cumulative calculations
        window_spec = Window.partitionBy("Team").orderBy("Season").rowsBetween(
            Window.unboundedPreceding, -1)

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
                        ).drop("row_num")

        team_data = team_data.select("Team", "Season", "Cumulative Won", "Cumulative Lost",
                                     "Cumulative Tied", "Cumulative W/L", "Cumulative AveRPW", "Cumulative AveRPO")

        # Add print statements to display columns and count
        print("Transformed team data columns:", team_data.columns)
        print("Transformed team data count:", team_data.count())

        # Save processed data
        utils.spark_save_data(team_data, config.PROCESSED_DATA_DIR, 'team_stats.csv')

    except Exception as e:
        logging.error(f"Error in preprocess_team_data task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")


if __name__ == "__main__":
    preprocess_team_data()