import os
import pandas as pd
from pyspark.sql import SparkSession

# create a SparkSession
spark = (
    SparkSession.builder.appName("CricketPrediction")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .getOrCreate()
)

# HDFS configuration
HDFS_HOST = os.getenv("HDFS_HOST", "192.168.245.142")
NAMENODE_PORT = os.getenv("NAMENODE_PORT", "8020")

# Load data
balltoball = spark.read.csv(
    f"hdfs://{HDFS_HOST}:{NAMENODE_PORT}/usr/ravi/t20/data/4_filteredData/ball_to_ball.csv",
    header=True,
    inferSchema=True,
)
team12_stats = spark.read.csv(
    f"hdfs://{HDFS_HOST}:{NAMENODE_PORT}/usr/ravi/t20/data/4_filteredData/team12_stats.csv",
    header=True,
    inferSchema=True,
)
players_stats = spark.read.csv(
    f"hdfs://{HDFS_HOST}:{NAMENODE_PORT}/usr/ravi/t20/data/4_filteredData/players_stats.csv",
    header=True,
    inferSchema=True,
)

import pandas as pd
import os

modeling = os.path.join("..", "ml_modeling")

# Convert and save balltoball dataset
balltoball_data = balltoball.collect()
balltoball_df = pd.DataFrame(balltoball_data, columns=balltoball.columns)
balltoball_df.to_csv(
    os.path.join(modeling, "data", "filtered_data", "balltoball.csv"), index=False
)
print("balltoball data saved")

# Convert and save team12_stats dataset
team12_stats_data = team12_stats.collect()
team12_stats_df = pd.DataFrame(team12_stats_data, columns=team12_stats.columns)
team12_stats_df.to_csv(
    os.path.join(modeling, "data", "filtered_data", "team12_stats.csv"), index=False
)
print("team12_stats data saved")

# Convert and save players_stats dataset
players_stats_data = players_stats.collect()
players_stats_df = pd.DataFrame(players_stats_data, columns=players_stats.columns)
players_stats_df.to_csv(
    os.path.join(modeling, "data", "filtered_data", "players_stats.csv"), index=False
)
print("players_stats data saved")
