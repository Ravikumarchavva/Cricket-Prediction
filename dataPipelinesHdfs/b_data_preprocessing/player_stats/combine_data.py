"""Combine batting, bowling, and fielding statistics into a single dataset."""

import os
import sys
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import config
from utils import create_spark_session, load_data, spark_save_data

def combine_data():
    """Combine batting, bowling, and fielding statistics into a single dataset."""
    logging.info("Starting combine_data task.")
    spark = create_spark_session("CombinePlayerStats")

    try:
        batting_data = load_data(spark, config.PROCESSED_DATA_DIR, 'batting_data.csv')
        bowling_data = load_data(spark, config.PROCESSED_DATA_DIR, 'bowling_data.csv')
        fielding_data = load_data(spark, config.PROCESSED_DATA_DIR, 'fielding_data.csv')
        players_data = load_data(spark, config.PROCESSED_DATA_DIR, 'match_players.csv').withColumnRenamed("country", "Country").withColumnRenamed("season", "Season").withColumnRenamed("player", "Player")

        # Data quality checks
        datasets = {
            "batting_data": batting_data,
            "bowling_data": bowling_data,
            "fielding_data": fielding_data,
            "players_data": players_data
        }
        for name, df in datasets.items():
            if df is None or df.count() == 0:
                logging.error(f"{name} is empty.")
                raise ValueError(f"{name} is empty.")
            
        print("Batting data columns:", batting_data.columns)
        print("Bowling data columns:", bowling_data.columns)
        print("Fielding data columns:", fielding_data.columns)
        print("Players data columns:", players_data.columns)

        required_columns = {
            "batting_data": ['Player', 'Country', 'Season', 'Cum Mat Total', 'Cum Inns Total', 'Cum Runs Total', 'Cum Batting Ave', 'Cum SR'],
            "bowling_data": ['Player', 'Country', 'Season', 'Cumulative Mat', 'Cumulative Inns', 'Cumulative Overs', 'Cumulative Runs', 'Cumulative Wkts', 'Cumulative Econ'],
            "fielding_data": ['Player', 'Country', 'Season', 'Cumulative Mat', 'Cumulative Inns', 'Cumulative Dis', 'Cumulative Ct', 'Cumulative St', 'Cumulative D/I'],
            "players_data": ["Player", "Country", "Season", "player_id"]
        }
        for name, cols in required_columns.items():
            missing_cols = [col for col in cols if col not in datasets[name].columns]
            if missing_cols:
                logging.error(f"Missing columns in {name}: {missing_cols}")
                raise ValueError(f"Missing columns in {name}: {missing_cols}")

        batting_data = batting_data.join(players_data, ['Player', 'Country', 'Season'], 'inner')
        bowling_data = bowling_data.join(players_data, ['Player', 'Country', 'Season'], 'inner')
        fielding_data = fielding_data.join(players_data, ['Player', 'Country', 'Season'], 'inner')

        batting_data = batting_data.select(['player_id', 'Player', 'Country', "Season","Cum Mat Total", "Cum Runs Total", 'Cum SR']).sort("Player","Season")
        bowling_data = bowling_data.select(['player_id', 'Player', 'Country', "Season","Cumulative Mat", "Cumulative Inns", 'Cumulative Overs','Cumulative Runs','Cumulative Wkts','Cumulative Econ']).withColumnRenamed("Cumulative Runs","Cumulative Bowling Runs")
        fielding_data = fielding_data.select(['player_id', 'Player', 'Country', "Season","Cumulative Mat", "Cumulative Inns", 'Cumulative Dis','Cumulative Ct','Cumulative St','Cumulative D/I'])
        print(batting_data.count(), bowling_data.count(), fielding_data.count())
        print(batting_data.columns, bowling_data.columns, fielding_data.columns)

        player_data = batting_data.join(bowling_data, on=['player_id','Player',"Country","Season"], how='inner').join(fielding_data, on=['player_id','Player',"Country","Season"], how='inner')\
                        .drop('Cumulative Mat','Cumulative Inns')

        # Add print statements to display columns and count
        print("Combined player data columns:", player_data.columns)
        print("Combined player data count:", player_data.count())

        spark_save_data(player_data, config.PROCESSED_DATA_DIR, 'player_stats.csv')
        logging.info("Data combining and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in combine_data task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

if __name__ == '__main__':
    combine_data()