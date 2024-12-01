from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
import requests
import io
import zipfile
import logging

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..','..'))
from configs import spark_config as config
from utils import spark_utils as utils

# Configure logging
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(message)s'
)

def upload_files_to_hdfs(client, hdfs_path, files):
    """Upload multiple files to HDFS using HDFSHook."""
    hdfs_files_set = utils.hdfs_list(client, hdfs_path)
    files_to_upload = {name: data for name, data in files.items() if name not in hdfs_files_set}

    if not files_to_upload:
        logging.info("All files are already present in HDFS.")
    else:
        print("Starting upload of files to HDFS.")
        with ThreadPoolExecutor(max_workers=100) as executor:  # Increased concurrency
            futures = [
                executor.submit(client.write, os.path.join(hdfs_path, name), io.BytesIO(data), overwrite=True)
                for name, data in files_to_upload.items()
            ]
            with tqdm(total=len(futures)) as pbar:
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)
        print("Finished uploading files to HDFS.")

def download_cricsheet():
    """Download a ZIP file, extract its contents, upload files to HDFS, and clean up locally."""
    try:
        logging.info("Starting download of ZIP file.")
        # Step 1: Download and extract the zip file concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            response_future = executor.submit(requests.get, 'https://cricsheet.org/downloads/t20s_csv2.zip')
            raw_data = io.BytesIO(response_future.result().content)
            extracted_data = zipfile.ZipFile(raw_data)
        logging.info("Successfully downloaded and extracted ZIP file.")
    except Exception as e:
        logging.error(f"Error downloading or extracting ZIP file: {e}")
        raise

    try:
        logging.info("Initializing Airflow HDFS client.")
        # Step 2: Initialize HDFS client
        client = utils.get_hdfs_client(id='webhdfs_default')  # Uses updated host and port
        logging.info("Successfully initialized HDFS client.")
        utils.ensure_hdfs_directory(client, os.path.join(config.RAW_DATA_DIR, 't20s_csv2'))

        # Collect all file names and data
        all_files = {
            os.path.basename(name): extracted_data.read(name)
            for name in extracted_data.namelist() if name.endswith('.csv')
        }

        # Upload files in parallel
        upload_files_to_hdfs(client, os.path.join(config.RAW_DATA_DIR, 't20s_csv2'), all_files)

        # Upload people.csv directly to HDFS
        logging.info("Downloading and uploading people.csv to HDFS.")
        people_response = requests.get('https://cricsheet.org/register/people.csv')
        people_response.raise_for_status()
        people_hdfs_path = os.path.join(config.RAW_DATA_DIR, 'people.csv')
        client.write(people_hdfs_path, io.BytesIO(people_response.content), overwrite=True)
        logging.info("Successfully uploaded people.csv to HDFS.")
    except Exception as e:
        logging.error(f"Error uploading files to HDFS: {e}")
        raise

    return


if __name__ == "__main__":
    download_cricsheet()