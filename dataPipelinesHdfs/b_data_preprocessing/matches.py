"""Module for preprocessing T20 cricket match data from raw files into structured format."""

import logging
import pandas as pd
import os
import sys
import concurrent.futures
from tqdm import tqdm

# Add the parent directory to sys.path for importing config and utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
import utils
import io

def process_single_match(client, match_id):
    """
    Process a single match file and return the processed DataFrame or None on failure.

    Args:
        client: The HDFS client for file operations.
        match: The match ID.
        match_id: Unique ID for each match.

    Returns:
        DataFrame of the processed match or None if an error occurs.
    """
    try:
        with client.read(os.path.join(config.RAW_DATA_DIR, 't20s_csv2', f'{match_id}_info.csv')) as reader:
            data = reader.read()
        match_df = pd.read_csv(io.StringIO(data.decode('utf-8')), header=None, names=['col1', 'attributes', 'values', 'players', 'code'])
        match_df = match_df.drop(columns=['col1', 'players', 'code']).T
        match_df.columns = match_df.iloc[0]
        match_df['match_id'] = match_id
        match_df = match_df[['match_id', 'team', 'team', 'gender', 'season', 'winner']].drop('attributes')
        match_df = match_df.reset_index(drop=True)
        return match_df
    except Exception as e:
        logging.warning(f"Error processing match {match_id}: {e}")
        return None

def preprocess_matches():
    """
    Process raw match data files and create a structured matches dataset.

    Returns:
        None
    """
    # Initialize logging
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Initialize HDFS client
        client = utils.get_hdfs_client()

        # List contents of HDFS directory
        dir_contents = utils.hdfs_list(client, os.path.join(config.RAW_DATA_DIR, 't20s_csv2'))
        info_files = [f for f in dir_contents if f.endswith('_info.csv')]
        match_ids = [f.split('_')[0] for f in info_files]

        if not info_files:
            logging.warning(f"No info files found in {os.path.join(config.RAW_DATA_DIR, 't20s_csv2')}.")
            return

        recalculated_matches = []
        injured_matches = []

        # Process each match concurrently using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_single_match, client, match_id): match_id
                for match_id in match_ids
            }

            # Gather results with tqdm for progress tracking
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Matches"):
                match = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        recalculated_matches.append(result)
                    else:
                        injured_matches.append(match)
                except Exception as e:
                    logging.error(f"Error in future for match {match}: {e}")
                    injured_matches.append(match)

        if recalculated_matches:
            # Concatenate all match DataFrames
            matches_data = pd.concat(recalculated_matches, ignore_index=True)
            matches_data.columns = ['match_id', 'team1', 'team2', 'team1_duplicate', 'team2_duplicate', 'gender', 'season', 'winner']
            matches_data = matches_data.drop(columns=['team1_duplicate', 'team2_duplicate'])

            # Save matches_data directly to HDFS
            utils.ensure_hdfs_directory(client, config.PROCESSED_DATA_DIR)
            matches_csv_path = os.path.join(config.PROCESSED_DATA_DIR, 'matches.csv')
            csv_data = matches_data.to_csv(index=False)
            client.write(matches_csv_path, data=csv_data, overwrite=True)
            print('Matches data processing and saving completed successfully.')
        else:
            logging.critical('No matches were successfully processed.')
            raise Exception('No matches were successfully processed.')

        logging.info(f'Successfully processed matches: {len(recalculated_matches)}')
        logging.info(f'Failed matches: {len(injured_matches)}')

    except Exception as e:
        logging.critical(f'Critical error: {e}')
        raise

if __name__ == '__main__':
    preprocess_matches()
