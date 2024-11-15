"""Preprocessing utilities for cricket player statistics data."""

import logging
from typing import Dict
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    when, col, regexp_extract, sum as spark_sum, row_number, round
)
from pyspark.sql import Window


def preprocess_batting_data(batting_data: DataFrame) -> DataFrame:
    """
    Clean and preprocess batting statistics data.

    Args:
        batting_data: Spark DataFrame containing raw batting statistics.

    Returns:
        Spark DataFrame with cleaned and processed batting statistics.
    """
    # Data quality checks
    required_columns = ["Player", "Season", "Mat", "Inns", "Runs", "SR", "Ave"]
    missing_columns = [col for col in required_columns if col not in batting_data.columns]
    if missing_columns:
        logging.error(f"Missing columns in batting data: {missing_columns}")
        raise ValueError(f"Missing columns: {missing_columns}")
    if batting_data.count() == 0:
        logging.error("Batting data has no rows.")
        raise ValueError("Batting data is empty.")

    logging.info("Preprocessing batting data.")
    batting_data = batting_data.select(
        "Player", "Season", "Mat", "Inns", "Runs", "SR", "Ave"
    ).sort("Player", "Season")

    batting_data = batting_data.withColumn(
        "Inns",
        when(col("Inns") == "-", "0").otherwise(col("Inns")).cast("int")
    ).withColumn(
        "Runs",
        when(col("Runs") == "-", "0").otherwise(col("Runs")).cast("int")
    ).withColumn(
        "SR",
        when(col("SR") == "-", "0").otherwise(col("SR")).cast("float")
    ).withColumn(
        "Ave",
        when(col("Ave") == "-", col("Runs") / col("Inns"))
        .otherwise(col("Ave")).cast("float")
    ).fillna({'Ave': 0})
    
    # Extract 'Country' and 'player_id' from 'Player' column
    batting_data = batting_data.withColumn(
        "Country",
        regexp_extract(col("Player"), r"\((.*?)\)", 1)
    ).withColumn(
        "player_id",
        regexp_extract(col("Player"), r"\[(.*?)\]", 1)
    ).withColumn(
        "Player",
        regexp_extract(col("Player"), r"^(.*?)\s\(", 1)
    )

    window_spec = Window.partitionBy("Player", "Country").orderBy("Season").rowsBetween(Window.unboundedPreceding, -1)
    row_num_window = Window.partitionBy("Player", "Country").orderBy("Season")

    # Define a window for cumulative calculations up to the previous season
    window_spec = Window.partitionBy("Player", "Country").orderBy("Season").rowsBetween(Window.unboundedPreceding, -1)

    # Window for row number to identify the first row per player and country
    row_num_window = Window.partitionBy("Player", "Country").orderBy("Season")

    # Calculate cumulative metrics excluding the current season and set to 0 if it's the first row
    batting_data = batting_data.withColumn("row_num", row_number().over(row_num_window)) \
        .withColumn("Cum Mat Total", 
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("Mat").over(window_spec))) \
        .withColumn("Cum Inns Total", 
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("Inns").over(window_spec))) \
        .withColumn("Cum Runs Total", 
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("Runs").over(window_spec))) \
        .withColumn("Cum Batting Ave", 
                    when(col("row_num") == 1, 0)
                    .otherwise(
                        round(when(spark_sum("Inns").over(window_spec) != 0,
                                    spark_sum(col("Inns") * col("Ave")).over(window_spec) / spark_sum("Inns").over(window_spec))
                                .otherwise(0), 2))) \
        .withColumn("Cum SR", 
                    when(col("row_num") == 1, 0)
                    .otherwise(
                        round(when(spark_sum("Inns").over(window_spec) != 0,
                                    spark_sum(col("Inns") * col("SR")).over(window_spec) / spark_sum("Inns").over(window_spec))
                                .otherwise(0), 2))) \
        .drop("row_num")

    # Include 'player_id' in the selected columns
    batting_data = batting_data.select(
        [ 'Player', 'Country', 'Season', 'Cum Mat Total',
         'Cum Inns Total', 'Cum Runs Total', 'Cum Batting Ave', 'Cum SR']
    )
    return batting_data

def preprocess_bowling_data(bowling_data: DataFrame) -> DataFrame:
    """
    Clean and preprocess bowling statistics data.

    Args:
        bowling_data: Spark DataFrame containing raw bowling statistics.

    Returns:
        Spark DataFrame with cleaned and processed bowling statistics.
    """
    # Data quality checks
    required_columns = ["Player", "Season", "Mat", "Inns", "Overs", "Runs", "Wkts", "Econ"]
    missing_columns = [col for col in required_columns if col not in bowling_data.columns]
    if missing_columns:
        logging.error(f"Missing columns in bowling data: {missing_columns}")
        raise ValueError(f"Missing columns: {missing_columns}")
    if bowling_data.count() == 0:
        logging.error("Bowling data has no rows.")
        raise ValueError("Bowling data is empty.")

    logging.info("Preprocessing bowling data.")
    bowling_data = bowling_data.select(
        "Player", "Season", "Mat", "Inns", 'Overs', "Runs", "Wkts", "Econ"
    ).sort("Player", "Season")

    bowling_data = bowling_data.withColumn("Overs", when(col("Overs") == "-", "0").otherwise(col("Overs")).cast("float"))
    bowling_data = bowling_data.withColumn("Wkts", when(col("Wkts") == "-", "0").otherwise(col("Wkts")).cast("float"))
    bowling_data = bowling_data.withColumn("Inns", when(col("Inns") == "-", "0").otherwise(col("Inns")).cast("float"))
    bowling_data = bowling_data.withColumn("Runs", when(col("Runs") == "-", "0").otherwise(col("Runs")).cast("float"))
    bowling_data = bowling_data.withColumn("Econ", when(col("Econ") == "-", col("Runs")/col("Inns")).otherwise(col("Econ")).cast("float")).na.fill(0)

    # Extract 'Country' and 'player_id' from 'Player' column
    bowling_data = bowling_data.withColumn(
        "Country",
        regexp_extract(col("Player"), r"\((.*?)\)", 1)
    ).withColumn(
        "player_id",
        regexp_extract(col("Player"), r"\[(.*?)\]", 1)
    ).withColumn(
        "Player",
        regexp_extract(col("Player"), r"^(.*?)\s\(", 1)
    )

    window_spec = Window.partitionBy("Player", "Country").orderBy("Season").rowsBetween(Window.unboundedPreceding, -1)
    row_num_window = Window.partitionBy("Player", "Country").orderBy("Season")

    bowling_data = bowling_data.withColumn("row_num", row_number().over(row_num_window)) \
        .withColumn("Cumulative Mat",
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("Mat").over(window_spec))) \
        .withColumn("Cumulative Inns",
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("Inns").over(window_spec))) \
        .withColumn("Cumulative Overs",
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("Overs").over(window_spec))) \
        .withColumn("Cumulative Runs",
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("Runs").over(window_spec))) \
        .withColumn("Cumulative Wkts",
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("Wkts").over(window_spec))) \
        .withColumn(
            "Cumulative Econ",
            when(col("row_num") == 1, 0)
            .otherwise(
                round(
                    when(spark_sum("Inns").over(window_spec) != 0,
                         spark_sum(col("Inns")*col("Econ")).over(window_spec) / spark_sum("Inns").over(window_spec))
                    .otherwise(0), 2)
            )
        ).drop("row_num")

    # Include 'player_id' in the selected columns
    bowling_data = bowling_data.select(
        [ 'Player', 'Country', 'Season', 'Cumulative Mat',
         'Cumulative Inns', 'Cumulative Overs', 'Cumulative Runs',
         'Cumulative Wkts', 'Cumulative Econ']
    )

    return bowling_data

def preprocess_fielding_data(fielding_data: DataFrame) -> DataFrame:
    """
    Clean and preprocess fielding statistics data.

    Args:
        fielding_data: Spark DataFrame containing raw fielding statistics.

    Returns:
        Spark DataFrame with cleaned and processed fielding statistics.
    """
    # Data quality checks
    required_columns = ["Player", "Mat", "Inns", "Dis", "Ct", "St", "D/I", "Season"]
    missing_columns = [col for col in required_columns if col not in fielding_data.columns]
    if missing_columns:
        logging.error(f"Missing columns in fielding data: {missing_columns}")
        raise ValueError(f"Missing columns: {missing_columns}")
    if fielding_data.count() == 0:
        logging.error("Fielding data has no rows.")
        raise ValueError("Fielding data is empty.")

    logging.info("Preprocessing fielding data.")
    fielding_data = fielding_data.select(
        ['Player', "Mat", "Inns", "Dis", "Ct", "St", "D/I", "Season"]
    ).sort(["Player", "Season"])

    fielding_data = fielding_data.withColumn('Inns', when(col('Inns') == '-', '0').otherwise(col('Inns')).cast('float'))
    fielding_data = fielding_data.withColumn('Dis', when(col('Dis') == '-', '0').otherwise(col('Dis')).cast('float'))
    fielding_data = fielding_data.withColumn('Ct', when(col('Ct') == '-', '0').otherwise(col('Ct')).cast('float'))
    fielding_data = fielding_data.withColumn('St', when(col('St') == '-', '0').otherwise(col('St')).cast('float'))
    fielding_data = fielding_data.withColumn('D/I', when(col('D/I') == '-', col('Dis')/col('Inns')).otherwise(col('D/I')).cast('float')).na.fill(0)

    # Extract 'Country' and 'player_id' from 'Player' column
    fielding_data = fielding_data.withColumn(
        "Country",
        regexp_extract(col("Player"), r"\((.*?)\)", 1)
    ).withColumn(
        "player_id",
        regexp_extract(col("Player"), r"\[(.*?)\]", 1)
    ).withColumn(
        "Player",
        regexp_extract(col("Player"), r"^(.*?)\s\(", 1)
    )

    window_spec = Window.partitionBy("Player", "Country").orderBy("Season").rowsBetween(Window.unboundedPreceding, -1)
    row_num_window = Window.partitionBy("Player", "Country").orderBy("Season")

    fielding_data = fielding_data.withColumn("row_num", row_number().over(row_num_window)) \
        .withColumn("Cumulative Mat",
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("Mat").over(window_spec))) \
        .withColumn("Cumulative Inns",
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("Inns").over(window_spec))) \
        .withColumn("Cumulative Dis",
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("Dis").over(window_spec))) \
        .withColumn("Cumulative Ct",
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("Ct").over(window_spec))) \
        .withColumn("Cumulative St",
                    when(col("row_num") == 1, 0)
                    .otherwise(spark_sum("St").over(window_spec))) \
        .withColumn("Cumulative D/I",
                    when(col("row_num") == 1, 0)
                    .otherwise(
                        round(
                            when(spark_sum("Inns").over(window_spec) != 0,
                                 spark_sum(("Dis")).over(window_spec) / spark_sum("Inns").over(window_spec))
                            .otherwise(0), 2)
                    )
        ).drop("row_num")

    # Include 'player_id' in the selected columns
    fielding_data = fielding_data.select(
        ['Player', 'Country', 'Season', 'Cumulative Mat', 'Cumulative Inns', 'Cumulative Dis', 'Cumulative Ct', 'Cumulative St', 'Cumulative D/I']
    )

    return fielding_data

def map_country_codes(df: DataFrame, country_codes: Dict[str, str]) -> DataFrame:
    """
    Map country codes to full country names and filter data.

    Args:
        df: Spark DataFrame containing country codes.
        country_codes: Dictionary mapping country codes to full names.

    Returns:
        Spark DataFrame with mapped country names.
    """
    # Data quality checks
    if 'Country' not in df.columns:
        logging.error("Column 'Country' is missing in the DataFrame.")
        raise ValueError("Required column 'Country' is missing.")
    if df.count() == 0:
        logging.error("DataFrame has no rows.")
        raise ValueError("Input DataFrame is empty.")

    logging.info("Mapping country codes to full country names.")
    df = df.filter(col('Country').isin(list(country_codes.keys())))
    df = df.replace(country_codes, subset=['Country'])
    return df