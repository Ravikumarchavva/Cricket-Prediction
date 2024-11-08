import logging
from pyspark.sql.functions import when, col, regexp_extract, sum as spark_sum, row_number, round
from pyspark.sql import Window

def preprocess_batting_data(batting_data):
    """Clean and preprocess the batting data."""
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
    ).na.fill(0)

    batting_data = batting_data.withColumn(
        "Country",
        regexp_extract(col("Player"), r"\((.*?)\)", 1)
    ).withColumn(
        "Player",
        regexp_extract(col("Player"), r"^(.*?)\s\(", 1)
    )

    window_spec = Window.partitionBy("Player", "Country").orderBy("Season").rowsBetween(Window.unboundedPreceding, -1)
    row_num_window = Window.partitionBy("Player", "Country").orderBy("Season")

    batting_data = batting_data.withColumn("row_num", row_number().over(row_num_window)) \
        .withColumn(
            "Cum Mat Total",
            when(col("row_num") == 1, 0).otherwise(spark_sum("Mat").over(window_spec))
        ).withColumn(
            "Cum Inns Total",
            when(col("row_num") == 1, 0).otherwise(spark_sum("Inns").over(window_spec))
        ).withColumn(
            "Cum Runs Total",
            when(col("row_num") == 1, 0).otherwise(spark_sum("Runs").over(window_spec))
        ).withColumn(
            "Cum Batting Ave",
            when(col("row_num") == 1, 0).otherwise(
                round(
                    when(spark_sum("Inns").over(window_spec) != 0,
                         spark_sum(col("Inns") * col("Ave")).over(window_spec) / spark_sum("Inns").over(window_spec))
                    .otherwise(0), 2
                )
            )
        ).withColumn(
            "Cum SR",
            when(col("row_num") == 1, 0).otherwise(
                round(
                    when(spark_sum("Inns").over(window_spec) != 0,
                         spark_sum(col("Inns") * col("SR")).over(window_spec) / spark_sum("Inns").over(window_spec))
                    .otherwise(0), 2
                )
            )
        ).drop("row_num")

    batting_data = batting_data.select(
        ['Player', 'Country', 'Season', 'Cum Mat Total', 'Cum Inns Total', 'Cum Runs Total', 'Cum Batting Ave', 'Cum SR']
    )

    return batting_data

def preprocess_bowling_data(bowling_data):
    """Clean and preprocess the bowling data."""
    logging.info("Preprocessing bowling data.")
    bowling_data = bowling_data.select(
        "Player", "Season", "Mat", "Inns", 'Overs', "Runs", "Wkts", "Econ"
    ).sort("Player", "Season")

    bowling_data = bowling_data.withColumn("Overs", when(col("Overs") == "-", "0").otherwise(col("Overs")).cast("float"))
    bowling_data = bowling_data.withColumn("Wkts", when(col("Wkts") == "-", "0").otherwise(col("Wkts")).cast("float"))
    bowling_data = bowling_data.withColumn("Inns", when(col("Inns") == "-", "0").otherwise(col("Inns")).cast("float"))
    bowling_data = bowling_data.withColumn("Runs", when(col("Runs") == "-", "0").otherwise(col("Runs")).cast("float"))
    bowling_data = bowling_data.withColumn("Econ", when(col("Econ") == "-", col("Runs")/col("Inns")).otherwise(col("Econ")).cast("float")).na.fill(0)

    bowling_data = bowling_data.withColumn(
        "Country",
        regexp_extract(col("Player"), r"\((.*?)\)", 1)
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

    bowling_data = bowling_data.select(
        ["Player", "Country", "Season", "Cumulative Mat", "Cumulative Inns", "Cumulative Overs",
         "Cumulative Runs", "Cumulative Wkts", "Cumulative Econ"]
    )

    return bowling_data

def preprocess_fielding_data(fielding_data):
    """Clean and preprocess the fielding data."""
    logging.info("Preprocessing fielding data.")
    fielding_data = fielding_data.select(
        ['Player', "Mat", "Inns", "Dis", "Ct", "St", "D/I", "Season"]
    ).sort(["Player", "Season"])

    fielding_data = fielding_data.withColumn('Inns', when(col('Inns') == '-', '0').otherwise(col('Inns')).cast('float'))
    fielding_data = fielding_data.withColumn('Dis', when(col('Dis') == '-', '0').otherwise(col('Dis')).cast('float'))
    fielding_data = fielding_data.withColumn('Ct', when(col('Ct') == '-', '0').otherwise(col('Ct')).cast('float'))
    fielding_data = fielding_data.withColumn('St', when(col('St') == '-', '0').otherwise(col('St')).cast('float'))
    fielding_data = fielding_data.withColumn('D/I', when(col('D/I') == '-', col('Dis')/col('Inns')).otherwise(col('D/I')).cast('float')).na.fill(0)

    fielding_data = fielding_data.withColumn(
        "Country",
        regexp_extract(col("Player"), r"\((.*?)\)", 1)
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

    fielding_data = fielding_data.select(
        ['Player', 'Country', 'Season', 'Cumulative Mat', 'Cumulative Inns', 'Cumulative Dis',
         'Cumulative Ct', 'Cumulative St', 'Cumulative D/I']
    )

    return fielding_data
