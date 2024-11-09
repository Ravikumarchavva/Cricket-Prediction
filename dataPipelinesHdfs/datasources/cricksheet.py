import requests
import zipfile
import io
import os
import shutil
from hdfs import InsecureClient
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('hdfs').setLevel(logging.ERROR)
logging.getLogger('hdfs.client').setLevel(logging.ERROR)

def download_cricsheet():
    """Download a ZIP file, extract its contents, upload files to HDFS, and clean up locally."""
    try:
        logging.info("Starting download of ZIP file.")
        # Step 1: Download and extract the zip file
        response = requests.get('https://cricsheet.org/downloads/t20s_csv2.zip')
        raw_data = io.BytesIO(response.content)
        extracted_data = zipfile.ZipFile(raw_data)
        local_extract_path = 'temp_extracted_data'
        extracted_data.extractall(local_extract_path)
        logging.info("Successfully downloaded and extracted ZIP file.")
    except Exception as e:
        logging.error(f"Error downloading or extracting ZIP file: {e}")
        return

    try:
        logging.info("Initializing HDFS client.")
        # Step 2: Initialize HDFS client
        client = InsecureClient('http://192.168.245.142:9870', user='ravikumar')
        hdfs_path = '/usr/ravi/t20/data/1_rawData/t20s_csv2'

        # Remove all files in the HDFS directory
        logging.info(f"Removing all files in HDFS directory: {hdfs_path}")
        client.delete(hdfs_path, recursive=True)
        client.makedirs(hdfs_path)

        # Step 3: Define a function for uploading a single file to HDFS
        def upload_file(local_file_path):
            """Upload a single file to HDFS."""
            try:
                file_name = os.path.basename(local_file_path)
                hdfs_file_path = os.path.join(hdfs_path, file_name)
                client.upload(hdfs_file_path, local_file_path, overwrite=True)
            except Exception as e:
                logging.error(f"Error uploading file {local_file_path} to HDFS: {e}")

        logging.info("Starting upload of files to HDFS.")

        # Collect all file paths
        all_files = []
        for root, dirs, files in os.walk(local_extract_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                all_files.append(local_file_path)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(upload_file, file) for file in all_files]
            with tqdm(total=len(futures)) as pbar:
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)
        logging.info("Finished uploading files to HDFS.")

        # Upload people.csv directly to HDFS
        logging.info("Downloading and uploading people.csv to HDFS.")
        people_response = requests.get('https://cricsheet.org/register/people.csv')
        people_response.raise_for_status()
        people_hdfs_path = os.path.join(hdfs_path, '..', 'people.csv')
        with client.write(people_hdfs_path, overwrite=True) as writer:
            writer.write(people_response.content)
        logging.info("Successfully uploaded people.csv to HDFS.")
    except Exception as e:
        logging.error(f"Error uploading files to HDFS: {e}")
    finally:
        try:
            logging.info("Cleaning up local extracted files.")
            # Step 5: Clean up local extracted files
            shutil.rmtree(local_extract_path)
            logging.info("Cleanup completed.")
        except Exception as e:
            logging.error(f"Error cleaning up local files: {e}")

    return

def main():
    download_cricsheet()

if __name__ == "__main__":
    main()