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
        spark = utils.create_spark_session()

        # Load data from HDFS using utils
        teams = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'teamStats.csv')
        matches = utils.load_data(spark, config.PROCESSED_DATA_DIR, 'matches.csv')
        
        # Map team names between datasets
        team_name_mapping = {
            'U.S.A.': 'United States of America',
            'U.A.E.': 'United Arab Emirates',
            'Czech Rep.': 'Czech Republic',
            'P.N.G.': 'Papua New Guinea',
            'Cayman': 'Cayman Islands'
        }
        
        # Filter unmatched teams
        tdt = teams.select("Team").distinct().rdd.map(lambda row: row.Team).collect()
        mdt = matches.select("team1").distinct().rdd.map(lambda row: row.team1).collect()
        
        unmatched_teams = [team for team in tdt if team not in mdt and team not in team_name_mapping]
        unmatched_teams += [team for team in mdt if team not in tdt and team not in team_name_mapping.values()]
        
        teams = teams.filter(~teams.Team.isin(unmatched_teams))
        matches = matches.filter(~matches.team1.isin(unmatched_teams)).filter(~matches.team2.isin(unmatched_teams))
        
        teams = teams.replace(team_name_mapping, subset='Team')
        matches = matches.replace(team_name_mapping, subset=['team1', 'team2'])
        
        # Create flipped matches data
        # ...existing code...
        matches1 = matches.withColumn('flip', F.lit(0))
        matches2 = matches.withColumnRenamed('team1', 'temp_team') \
                          .withColumnRenamed('team2', 'team1') \
                          .withColumnRenamed('temp_team', 'team2') \
                          .withColumn('flip', F.lit(1))
        matchesflip = matches1.union(matches2).sort('match_id')
        
        # Join with team statistics
        # ...existing code...
        matchesflip = matchesflip.join(teams, on=['team1', 'season'], how='inner') \
                               .drop("Team", "season") \
                               .withColumnRenamed("Cumulative Won", "Cumulative Won team1")
        # ...existing code continues for team2...
        
        # Prepare final dataset
        # ...existing code...
        matchesflip = matchesflip.select(
            "match_id", "flip", "gender", "Cumulative Won team1", "Cumulative Lost team1",
            "Cumulative Tied team1", "Cumulative NR team1", "Cumulative W/L team1",
            "Cumulative AveRPW team1", "Cumulative AveRPO team1", "Cumulative Won team2",
            "Cumulative Lost team2", "Cumulative Tied team2", "Cumulative NR team2",
            "Cumulative W/L team2", "Cumulative AveRPW team2", "Cumulative AveRPO team2"
        ).sort("match_id", 'flip')
        
        # Save the processed data to HDFS using utils
        utils.save_data(matchesflip, config.MERGED_DATA_DIR, 'team12_stats_flip.csv')
        logging.info('Match team stats data saved successfully.')
        
    except Exception as e:
        logging.error(f"Error in processing match team stats: {e}")
        raise

if __name__ == '__main__':
    process_match_team_stats()




