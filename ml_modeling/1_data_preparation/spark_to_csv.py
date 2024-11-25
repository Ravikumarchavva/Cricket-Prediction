import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.getcwd(), '..','..'))
from configs import spark_config as config
from utils import spark_utils as utils

# create a SparkSession
spark = utils.create_spark_session("Data Preparation")

# Load data
balltoball = spark.read.csv(
    f"{config.HDFS_NAMENODE}/usr/ravi/t20/data/4_filteredData/ball_to_ball.csv",
    header=True,
    inferSchema=True,
)
team12_stats = spark.read.csv(
    f"{config.HDFS_NAMENODE}/usr/ravi/t20/data/4_filteredData/team12_stats.csv",
    header=True,
    inferSchema=True,
)
players_stats = spark.read.csv(
    f"{config.HDFS_NAMENODE}/usr/ravi/t20/data/4_filteredData/players_stats.csv",
    header=True,
    inferSchema=True,
)


modeling = os.path.join(os.path.dirname(__file__), "..")


# Convert and save balltoball dataset
balltoball_data = balltoball.collect()
balltoball_df = pd.DataFrame(balltoball_data, columns=balltoball.columns)
balltoball_df.to_csv(
    os.path.join(modeling, "filtered_data", "balltoball.csv"), index=False
)
print("balltoball data saved")

# Convert and save team12_stats dataset
team12_stats_data = team12_stats.collect()
team12_stats_df = pd.DataFrame(team12_stats_data, columns=team12_stats.columns)
team12_stats_df.to_csv(
    os.path.join(modeling, "filtered_data", "team12_stats.csv"), index=False
)
print("team12_stats data saved")

# Convert and save players_stats dataset
players_stats_data = players_stats.collect()
players_stats_df = pd.DataFrame(players_stats_data, columns=players_stats.columns)
players_stats_df.to_csv(
    os.path.join(modeling, "filtered_data", "players_stats.csv"), index=False
)
print("players_stats data saved")
