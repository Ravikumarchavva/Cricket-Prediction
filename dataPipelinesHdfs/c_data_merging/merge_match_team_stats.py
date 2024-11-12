"""Module for processing and merging match and team statistics data.

Integrates team statistics with match data, handles team name mappings, and prepares 
data for modeling. Uses configurations and utilities for Spark and HDFS operations.
"""

import os
import logging
from pyspark.sql import functions as F

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
import utils

def process_match_team_stats():
    """Merge match data with team statistics."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Initialize Spark session using utils
        spark = utils.create_spark_session("MatchTeamStats")

        # Load data from HDFS using utils
        teams = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'team_stats.csv')
        matches = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'matches.csv')

        # Identify distinct teams
        tdt = [row['Team'] for row in teams.select("Team").distinct().collect()]
        mdt = [row['team1'] for row in matches.select("team1").distinct().collect()]

        # Define team name mappings
        team_name_mapping = {
            'U.S.A.': 'United States of America',
            'U.A.E.': 'United Arab Emirates',
            'Czech Rep.': 'Czech Republic',
            'P.N.G.': 'Papua New Guinea',
            'Cayman': 'Cayman Islands'
        }

        # Identify unmatched teams after mapping
        unmatched_tdt = [team for team in tdt if team not in mdt and team not in team_name_mapping]
        unmatched_mdt = [team for team in mdt if team not in tdt and team not in team_name_mapping.values()]

        # Combine unmatched teams
        unmatched_teams = unmatched_tdt + unmatched_mdt

        # Filter out unmatched teams
        teams = teams.filter(~teams.Team.isin(unmatched_teams))
        matches = matches.filter(~matches.team1.isin(unmatched_teams)).filter(~matches.team2.isin(unmatched_teams))

        # Replace team names based on mapping
        teams = teams.replace(team_name_mapping, subset='Team')
        matches = matches.replace(team_name_mapping, subset=['team1', 'team2'])

        # Rename 'Team' to 'team1' and 'Season' to 'Season' in 'teams'
        teams = teams.withColumnRenamed('Team', 'team1').withColumnRenamed('Season', 'Season')

        # Create flipped matches
        matches1 = matches.withColumn('flip', F.lit(0))
        matches2 = matches.withColumnRenamed('team1', 'temp_team') \
                          .withColumnRenamed('team2', 'team1') \
                          .withColumnRenamed('temp_team', 'team2') \
                          .withColumn('flip', F.lit(1))
        matchesflip = matches1.union(matches2).sort('match_id')
        print(matchesflip.show(2))
        print(teams.show(2))

        # Rename 'season' to 'Season' in 'matchesflip'
        matchesflip = matchesflip.withColumnRenamed('season', 'Season')

        # Join with team1 statistics
        matchesflip = matchesflip.join(teams, on=['team1', 'Season'], how='inner') \
                               .drop("team1") \
                               .withColumnRenamed("Cumulative Won", "Cumulative Won team1") \
                               .withColumnRenamed("Cumulative Lost", "Cumulative Lost team1") \
                               .withColumnRenamed("Cumulative Tied", "Cumulative Tied team1") \
                               .withColumnRenamed("Cumulative NR", "Cumulative NR team1") \
                               .withColumnRenamed("Cumulative W/L", "Cumulative W/L team1") \
                               .withColumnRenamed("Cumulative AveRPW", "Cumulative AveRPW team1") \
                               .withColumnRenamed("Cumulative AveRPO", "Cumulative AveRPO team1")

        # Prepare 'teams' DataFrame for joining with 'team2'
        teams_renamed = teams.withColumnRenamed('team1', 'team2')
        # Join with team2 statistics
        matchesflip = matchesflip.join(teams_renamed, on=['team2', 'Season'], how='inner') \
                               .drop("team2", "Season") \
                               .withColumnRenamed("Cumulative Won", "Cumulative Won team2") \
                               .withColumnRenamed("Cumulative Lost", "Cumulative Lost team2") \
                               .withColumnRenamed("Cumulative Tied", "Cumulative Tied team2") \
                               .withColumnRenamed("Cumulative NR", "Cumulative NR team2") \
                               .withColumnRenamed("Cumulative W/L", "Cumulative W/L team2") \
                               .withColumnRenamed("Cumulative AveRPW", "Cumulative AveRPW team2") \
                               .withColumnRenamed("Cumulative AveRPO", "Cumulative AveRPO team2")

        # Process gender column
        matchesflip = matchesflip.withColumn("gender", F.when(matchesflip['gender'] == "male", 0).otherwise(1).cast("int"))

        # Select and sort final columns
        matchesflip = matchesflip.select(
            "match_id", "flip", "gender",
            "Cumulative Won team1", "Cumulative Lost team1",
            "Cumulative Tied team1",
            "Cumulative W/L team1", "Cumulative AveRPW team1",
            "Cumulative AveRPO team1", "Cumulative Won team2",
            "Cumulative Lost team2", "Cumulative Tied team2",
             "Cumulative W/L team2",
            "Cumulative AveRPW team2", "Cumulative AveRPO team2"
        ).sort("match_id", 'flip')
        # Save the processed data to HDFS using utils
        utils.spark_save_data(matchesflip, config.MERGED_DATA_DIR, 'team12_stats_flip.csv')
        logging.info('Match team stats data saved successfully.')

    except Exception as e:
        logging.error(f"Error in processing match team stats: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

    return

if __name__ == '__main__':
    process_match_team_stats()




