#!/usr/bin/env python
# coding: utf-8

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    when, col, regexp_extract, sum as spark_sum, row_number, round, isnan, count
)
from pyspark.sql import Window

def create_spark_session():
    """Create and return a Spark session."""
    return SparkSession.builder.appName("PlayerStats").getOrCreate()

def load_data(spark, data_dir, filename):
    """Load CSV data."""
    return spark.read.csv(
        os.path.join(data_dir, filename),
        header=True,
        inferSchema=True
    )

def preprocess_batting_data(batting_data):
    """Clean and preprocess the batting data."""
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
        ["Player", "Country", "Season", "Cumulative Mat", "Cumulative Inns", "Cumulative Overs", "Cumulative Runs", "Cumulative Wkts", "Cumulative Econ"]
    )

    return bowling_data

def preprocess_fielding_data(fielding_data):
    """Clean and preprocess the fielding data."""
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
        ['Player', 'Country', 'Season', 'Cumulative Mat', 'Cumulative Inns', 'Cumulative Dis', 'Cumulative Ct', 'Cumulative St', 'Cumulative D/I']
    )

    return fielding_data

def map_country_codes(df, country_codes):
    """Map country codes to full country names and filter data."""
    df = df.filter(col('Country').isin(list(country_codes.keys())))
    df = df.replace(country_codes, subset=['Country'])
    return df

def load_players_data(spark, data_dir):
    """Load players data."""
    players_data = load_data(spark, data_dir, 'Players.csv')
    players_data = players_data.withColumnRenamed("player", "Player").withColumnRenamed("country", "Country")
    return players_data

def save_data(df, data_dir, filename):
    """Save DataFrame to CSV."""
    output_path = os.path.join(data_dir, filename)
    df.toPandas().to_csv(output_path, index=False)

def main():
    try:
        spark = create_spark_session()
        # Update base_dir to point to the project's root directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        raw_data_dir = os.path.join(base_dir, 'data', '1_rawData')
        processed_data_dir = os.path.join(base_dir, 'data', '2_processedData')

        # Define country codes mapping
        country_codes = {
            'LES': 'Lesotho', 'BUL': 'Bulgaria', 'VAN': 'Vanuatu', 'ROM': 'Romania', 'Aut': 'Austria', 'COK': 'Cook Islands', 'Fran': 'France', 'SRB': 'Serbia', 'PAK': 'Pakistan', 'HUN': 'Hungary', 'CYP': 'Cyprus', 'Fiji': 'Fiji', 'FIN': 'Finland', 'EST': 'Estonia', 'CHN': 'China', 'GRC': 'Greece', 'CAM': 'Cambodia', 'GUE': 'Guernsey', 'SEY': 'Seychelles', 'JPN': 'Japan', 'TAN': 'Tanzania', 'JER': 'Jersey', 'QAT': 'Qatar', 'ENG': 'England', 'UGA': 'Uganda', 'BER': 'Bermuda', 'CZK-R': 'Czech Republic', 'CAY': 'Cayman Islands', 'IRE': 'Ireland', 'Mali': 'Mali', 'BRA': 'Brazil', 'SUI': 'Switzerland', 'Peru': 'Peru', 'Mex': 'Mexico', 'MOZ': 'Mozambique', 'Samoa': 'Samoa', 'HKG': 'Hong Kong', 'BAN': 'Bangladesh', 'SL': 'Sri Lanka', 'PNG': 'Papua New Guinea', 'ZIM': 'Zimbabwe', 'GHA': 'Ghana', 'SWZ': 'Eswatini', # Swaziland's official name now is Eswatini
            'MYAN': 'Myanmar', 'IND': 'India', 'USA': 'United States of America', 'NEP': 'Nepal', 'AFG': 'Afghanistan', 'PAN': 'Panama', 'NGA': 'Nigeria', 'SLE': 'Sierra Leone', 'ESP': 'Spain', 'Bhm': 'Bahamas', 'TKY': 'Turkey', 'MWI': 'Malawi', 'WI': 'West Indies', 'IOM': 'Isle of Man', 'THA': 'Thailand', 'SWA': 'Eswatini',
            'SKOR': 'South Korea', 'GMB': 'Gambia', 'ISR': 'Israel', 'KUW': 'Kuwait', 'Belg': 'Belgium', 'GER': 'Germany', 'ITA': 'Italy', 'CAN': 'Canada', 'MDV': 'Maldives', 'Blz': 'Belize', 'DEN': 'Denmark', 'INA': 'Indonesia', 'KENYA': 'Kenya', 'LUX': 'Luxembourg', 'STHEL': 'Saint Helena', 'BHR': 'Bahrain', 'KSA': 'Saudi Arabia', 'MLT': 'Malta', 'Arg': 'Argentina', 'MNG': 'Mongolia', 'AUS': 'Australia', 'GIBR': 'Gibraltar', 'SGP': 'Singapore', 'Chile': 'Chile', 'UAE': 'United Arab Emirates', 'NZ': 'New Zealand', 'SCOT': 'Scotland', 'BHU': 'Bhutan', 'MAS': 'Malaysia', 'BOT': 'Botswana', 'CRC': 'Costa Rica', 'PHI': 'Philippines', 'NAM': 'Namibia', 'RWN': 'Rwanda', 'OMA': 'Oman', 'NOR': 'Norway', 'CRT': 'Croatia', 'SWE': 'Sweden', 'Iran': 'Iran', 'PORT': 'Portugal', 'NED': 'Netherlands', 'SA': 'South Africa', 'SVN': 'Slovenia', 'GUE': 'Guernsey', 'MDV': 'Maldives', 'BHM': 'Bahamas', 'SWE': 'Sweden', 'MLT': 'Malta', 'ITA': 'Italy',
        }

        # Load and preprocess batting data
        batting_data = load_data(spark, raw_data_dir, 't20_batting_stats.csv')
        batting_data = preprocess_batting_data(batting_data)
        batting_data = map_country_codes(batting_data, country_codes)
        
        # Load and preprocess bowling data
        bowling_data = load_data(spark, raw_data_dir, 't20_bowling_stats.csv')
        bowling_data = preprocess_bowling_data(bowling_data)
        bowling_data = map_country_codes(bowling_data, country_codes)
        
        # Load and preprocess fielding data
        fielding_data = load_data(spark, raw_data_dir, 't20_fielding_stats.csv')
        fielding_data = preprocess_fielding_data(fielding_data)
        fielding_data = map_country_codes(fielding_data, country_codes)

        # Load players data
        players_data = load_players_data(spark, processed_data_dir)

        # Join data with players data
        batting_data = batting_data.join(players_data, ['Player', 'Country'], 'inner')
        bowling_data = bowling_data.join(players_data, ['Player', 'Country'], 'inner')
        fielding_data = fielding_data.join(players_data, ['Player', 'Country'], 'inner')

        # Save the processed data to CSV
        save_data(batting_data, processed_data_dir, 'batting.csv')
        save_data(bowling_data, processed_data_dir, 'bowling.csv')
        save_data(fielding_data, processed_data_dir, 'fielding.csv')

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()




