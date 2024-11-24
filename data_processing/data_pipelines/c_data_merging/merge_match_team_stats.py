"""Module for processing and merging match and team statistics data.

Integrates team statistics with match data, handles team name mappings, and prepares 
data for modeling. Uses configurations and utilities for Spark and HDFS operations.
"""

import os
import logging
from pyspark.sql import functions as F

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
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
        matches = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'matches.csv').withColumnRenamed('season', 'Season')
        
        # Data quality checks and logging
        teams_rows, teams_cols = teams.count(), len(teams.columns)
        matches_rows, matches_cols = matches.count(), len(matches.columns)
        logging.info(f'Teams data: {teams_rows} rows, {teams_cols} columns')
        logging.info(f'Matches data: {matches_rows} rows, {matches_cols} columns')
        
        # Check for nulls in critical columns
        teams_nulls = teams.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in teams.columns])
        matches_nulls = matches.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in matches.columns])
        logging.info('Null values in teams data:')
        teams_nulls.show()
        logging.info('Null values in matches data:')
        matches_nulls.show()

        # Identify distinct teams
        tdt = teams.select("Team").distinct().collect()
        mdt = matches.select("winner").distinct().collect()

        # Define team name mappings
        team_name_mapping = {
            'U.S.A.': 'United States of America',
            'U.A.E.': 'United Arab Emirates',
            'Czech Rep.': 'Czech Republic',
            'P.N.G.': 'Papua New Guinea',
            'Cayman': 'Cayman Islands'
        }

        # Identify unmatched teams
        unmatched_tdt = [team["Team"] for team in tdt if team not in mdt and team not in team_name_mapping]
        unmatched_mdt = [team["winner"] for team in mdt if team not in tdt and team not in team_name_mapping.values()]
        unmatched_teams = unmatched_tdt + unmatched_mdt

        # Filter out unmatched teams
        teams = teams.filter(~teams['Team'].isin(unmatched_teams))
        matches = matches.filter(~matches['team1'].isin(unmatched_teams)).filter(~matches['team2'].isin(unmatched_teams))

        # Replace team names based on the mapping
        teams = teams.replace(team_name_mapping, subset='Team')
        matches = matches.replace(team_name_mapping, subset=['team1', 'team2', 'winner'])

        # Merge team statistics for team1
        matches = matches.withColumnRenamed("Season", "Match_Season")
        matches = matches.join(
            teams,
            (matches['team1'] == teams['Team']) & (matches['Match_Season'] == teams['Season']),
            how='inner'
        ).drop("Team", "Season")

        # Rename columns for team1
        for old_name, new_name in {
            "Cumulative Won": "Cumulative Won team1",
            "Cumulative Lost": "Cumulative Lost team1",
            "Cumulative Tied": "Cumulative Tied team1",
            "Cumulative W/L": "Cumulative W/L team1",
            "Cumulative AveRPW": "Cumulative AveRPW team1",
            "Cumulative AveRPO": "Cumulative AveRPO team1",
        }.items():
            matches = matches.withColumnRenamed(old_name, new_name)

        # Merge team statistics for team2
        matches = matches.join(
            teams,
            (matches['team2'] == teams['Team']) & (matches['Match_Season'] == teams['Season']),
            how='inner'
        ).drop("Team", "Match_Season")

        # Rename columns for team2
        for old_name, new_name in {
            "Cumulative Won": "Cumulative Won team2",
            "Cumulative Lost": "Cumulative Lost team2",
            "Cumulative Tied": "Cumulative Tied team2",
            "Cumulative W/L": "Cumulative W/L team2",
            "Cumulative AveRPW": "Cumulative AveRPW team2",
            "Cumulative AveRPO": "Cumulative AveRPO team2",
        }.items():
            matches = matches.withColumnRenamed(old_name, new_name)

        # Process gender column
        matches = matches.withColumn("gender", F.when(matches['gender'] == "male", 0).otherwise(1).cast("int"))

        # Select and sort final columns
        matches = matches.select(
            "match_id", "gender",
            "Cumulative Won team1", "Cumulative Lost team1",
            "Cumulative Tied team1", "Cumulative W/L team1",
            "Cumulative AveRPW team1", "Cumulative AveRPO team1",
            "Cumulative Won team2", "Cumulative Lost team2",
            "Cumulative Tied team2", "Cumulative W/L team2",
            "Cumulative AveRPW team2", "Cumulative AveRPO team2"
        ).sort("match_id")

        # Save the processed data to HDFS using utils
        utils.spark_save_data(matches, config.MERGED_DATA_DIR, 'team_stats.csv')
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




