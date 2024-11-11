"""Module for downloading and processing cricket data from Cricsheet."""

import io
import logging
import os
import sys
import zipfile
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
import utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def download_cricsheet():
    """Download a ZIP file, extract its contents, upload files to HDFS, and clean up locally."""
    try:
        print("Starting download of ZIP file.")
        response = requests.get('https://cricsheet.org/downloads/t20s_csv2.zip')
        response.raise_for_status()
        raw_data = io.BytesIO(response.content)
        extracted_data = zipfile.ZipFile(raw_data)
        logging.info("Successfully downloaded and extracted ZIP file.")
    except Exception as e:
        logging.error(f"Error downloading or extracting ZIP file: {e}")
        raise RuntimeError("Failed to download or extract ZIP file.") from e

    try:
        logging.info("Initializing HDFS client.")
        client = utils.get_hdfs_client()
        hdfs_path = os.path.join(config.RAW_DATA_DIR, 't20s_csv2')

        logging.info("Starting upload of files to HDFS.")
        all_files = {name: extracted_data.read(name) for name in extracted_data.namelist() if name.endswith('.csv')}
        utils.upload_files_to_hdfs(client, hdfs_path, all_files)

        logging.info("Downloading and uploading people.csv to HDFS.")
        people_response = requests.get('https://cricsheet.org/register/people.csv')
        people_response.raise_for_status()
        people_hdfs_path = os.path.join(config.RAW_DATA_DIR, 'people.csv')
        people_data = people_response.content.decode('utf-8')
        utils.hdfs_write(client, people_hdfs_path, data=people_data, overwrite=True)
        logging.info("Successfully uploaded people.csv to HDFS.")
    except Exception as e:
        logging.error(f"Error uploading files to HDFS: {e}")
        raise RuntimeError("Failed to upload files to HDFS.") from e

def main():
    """Execute the Cricsheet data download process."""
    """Execute the Cricsheet data download process."""
    download_cricsheet()


if __name__ == "__main__":
    main()

