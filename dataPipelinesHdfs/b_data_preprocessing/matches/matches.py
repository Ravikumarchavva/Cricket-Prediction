import os
import logging
import polars as pl
from hdfs import InsecureClient
import config
import tempfile
import io

def preprocess_matches():
    # Initialize logging
    logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Initialize HDFS client
        client = InsecureClient(f'http://{config.HDFS_HOST}:{config.HDFS_HTTP_PORT}', user=config.HDFS_USER)
        hdfs_data_path = config.HDFS_BASE_DIR

        # Check the contents of the directory on HDFS
        logging.info(f'Checking contents of HDFS directory: {os.path.join(hdfs_data_path, "1_rawData", "t20s_csv2")}')
        dir_contents = client.list(os.path.join(hdfs_data_path, '1_rawData', 't20s_csv2'))

        # Find all CSV files in the specified directory
        info_files = [f for f in dir_contents if f.endswith('_info.csv')]
        logging.info(f'Found {len(info_files)} info files.')

        if len(info_files) == 0:
            logging.warning(f'No info files found in {os.path.join(hdfs_data_path, "1_rawData", "t20s_csv2")}. Please check the directory and file permissions.')

        match_ids = [f.split('_')[0] for f in info_files]

        # Define the initial and final schemas
        initial_schema = {'col1': pl.Utf8, 'attributes': pl.Utf8, 'values': pl.Utf8, 'players': pl.Utf8, 'code': pl.Utf8}
        final_schema = [
            ('team1', pl.Utf8),
            ('team2', pl.Utf8),
            ('gender', pl.Utf8),
            ('season', pl.Utf8),
            ('date', pl.Utf8),
            ('venue', pl.Utf8),
            ('city', pl.Utf8),
            ('toss_winner', pl.Utf8),
            ('toss_decision', pl.Utf8),
            ('winner', pl.Utf8),
        ]

        # Create a dictionary from the final schema
        final_schema_dict = {key: value for key, value in final_schema}

        # Initialize an empty DataFrame with the final schema
        matches_data = pl.DataFrame(schema=final_schema_dict)

        # List to store recalculated match IDs
        recalculated_matchids = match_ids[:]
        from tqdm import tqdm
        # Iterate over matches and process each one
        for idx, match in tqdm(enumerate(info_files)):
            try:
                with client.read(os.path.join(hdfs_data_path, '1_rawData', 't20s_csv2', match)) as reader:
                    match_df = pl.read_csv(reader, schema=initial_schema)
                # Extract team names
                team1_name = match_df[1, 'values']
                team2_name = match_df[2, 'values']

                # Replace team names
                match_df = match_df.with_columns([
                    pl.when((pl.col('attributes') == 'team') & (pl.col('values') == team1_name))
                    .then(pl.lit('team1'))
                    .when((pl.col('attributes') == 'team') & (pl.col('values') == team2_name))
                    .then(pl.lit('team2'))
                    .otherwise(pl.col('attributes'))
                    .alias('attributes')
                ])

                # Select and transpose the DataFrame
                match_transposed = match_df.select("attributes", "values").transpose(include_header=True, column_names="attributes").drop("column")

                # Ensure all columns in final_schema_dict are present
                missing_cols = [col for col in final_schema_dict.keys() if col not in match_transposed.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns: {missing_cols}")

                # Select the required columns and append to matches_data
                match_transposed = match_transposed.select(final_schema_dict.keys())
                matches_data = matches_data.vstack(match_transposed)
            except Exception as e:
                recalculated_matchids.remove(match_ids[idx])

        matches_data = matches_data.with_columns(pl.Series(recalculated_matchids).alias("match_id").cast(pl.Int64))

        # Save matches_data to an in-memory buffer  -- need to optimize this
        buffer = io.BytesIO()
        matches_data.write_csv(buffer)
        buffer.seek(0)
        logging.info('Saved matches.csv to in-memory buffer.')

        # Upload the buffer content to HDFS
        try:
            client.write(f'{config.PROCESSED_DATA_DIR}/matches.csv', data=buffer, overwrite=True)
            print('Uploaded matches.csv to HDFS.')
        except Exception as e:
            logging.error(f'Error uploading matches.csv to HDFS: {e}')
            raise
    except Exception as e:
        logging.critical(f'Critical error: {e}')
        raise

if __name__ == '__main__':
    preprocess_matches()