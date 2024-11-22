"""Module for processing player information from T20 cricket matches."""

import os
import sys
import logging
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import config
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_players_data():
    """
    Process player information from match data files.

    This function reads player information from match info files, processes them
    to create two datasets:
    1. match_players - containing player participation in each match
    2. players - containing unique player records with their team information

    Returns:
        None
    """
    try:
        # Initialize HDFS client
        client = utils.get_hdfs_client()

        # Check the contents of the directory on HDFS
        logging.info(f'Checking contents of HDFS directory: {os.path.join(config.RAW_DATA_DIR, "t20s_csv2")}')
        dir_contents = utils.hdfs_list(client, os.path.join(config.RAW_DATA_DIR, 't20s_csv2'))

        # Find all CSV files in the specified directory
        info_files = [f for f in dir_contents if f.endswith('_info.csv')]
        logging.info(f'Found {len(info_files)} info files.')

        if len(info_files) == 0:
            logging.warning(f'No info files found in {os.path.join(config.RAW_DATA_DIR, "t20s_csv2")}. Please check the directory and file permissions.')

        dataframes = pd.DataFrame(columns=['country', 'player', 'player_id', 'season', 'match_id'])
        injured_matches = []
        from tqdm import tqdm
        for info_file in tqdm(info_files):
            match_id = pd.to_numeric(info_file.split('/')[-1].split('_')[0])
            try:
                with client.read(os.path.join(config.RAW_DATA_DIR, 't20s_csv2', info_file)) as reader:
                    df = pd.read_csv(reader, header=None, names=['type', 'heading', 'subkey', 'players', 'player_id'], skipinitialspace=True).drop('type', axis=1)
                players_df = df[df['heading'] == "player"].drop(['heading', 'player_id'], axis=1)
                registry_df = df[df['heading'] == "registry"].drop('heading', axis=1)
                merged_df = players_df.merge(registry_df[['players', 'player_id']], on='players', how='inner')
                merged_df.rename(columns={'players': 'player', 'subkey': 'country'}, inplace=True)
                season = df['subkey'][5]
                merged_df['match_id'] = match_id
                merged_df['season'] = season
                if len(merged_df) != 22:
                    raise Exception('Injured Match')
                dataframes = pd.concat([dataframes, merged_df])
            except Exception:
                injured_matches.append(match_id)

        logging.info(f'Processed all files. Injured matches: {injured_matches}')

        # Data quality checks for match_players DataFrame
        if dataframes.empty:
            logging.error("No player data extracted.")
            raise ValueError("Extracted player data is empty.")
        required_columns = ["country", "player", "player_id", "season", "match_id"]
        missing_columns = [col for col in required_columns if col not in dataframes.columns]
        if missing_columns:
            logging.error(f"Missing columns in player data: {missing_columns}")
            raise ValueError(f"Missing columns in player data: {missing_columns}")

        # Save dataframes to HDFS
        try:
            data = dataframes.to_csv(index=False)
            client.write(f'{config.PROCESSED_DATA_DIR}/match_players.csv', data=data, encoding='utf-8', overwrite=True)
            print('Saved match_players.csv to HDFS.')
        except Exception as e:
            logging.error(f'Error saving match_players to HDFS: {e}')
            raise

        # Individual player's data
        players = dataframes.drop('match_id', axis=1).drop_duplicates(subset=['player', 'country', 'player_id'])

        # Data quality checks for players DataFrame
        if players.empty:
            logging.error("No unique players data extracted.")
            raise ValueError("Extracted unique players data is empty.")

        # Save players to HDFS
        try:
            data = players.to_csv(index=False)
            client.write(f'{config.PROCESSED_DATA_DIR}/players.csv', data=data, encoding='utf-8', overwrite=True)
            print('Saved players.csv to HDFS.')
        except Exception as e:
            logging.error(f'Error saving players.csv to HDFS: {e}')
            raise

    except Exception as e:
        logging.critical(f'Critical error: {e}')
        raise
    return

if __name__ == "__main__":
    process_players_data()