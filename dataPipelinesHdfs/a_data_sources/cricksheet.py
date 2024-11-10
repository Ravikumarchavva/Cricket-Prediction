import requests
import zipfile
import io
import os
from hdfs import InsecureClient
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('hdfs').setLevel(logging.ERROR)
logging.getLogger('hdfs.client').setLevel(logging.ERROR)

def download_cricsheet():
    """Download a ZIP file, extract its contents, upload files to HDFS, and clean up locally."""
    try:
        logging.info("Starting download of ZIP file.")
        # Step 1: Download and extract the zip file
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
        # Step 2: Initialize HDFS client
        client = InsecureClient('http://192.168.245.142:9870', user='ravikumar')
        hdfs_path = '/usr/ravi/t20/data/1_rawData/t20s_csv2'

        # Step 3: Define a function for uploading a single file to HDFS
        def upload_file(file_name, file_data):
            """Upload a single file to HDFS."""
            try:
                hdfs_file_path = os.path.join(hdfs_path, file_name)
                with client.write(hdfs_file_path, overwrite=True) as writer:
                    writer.write(file_data)
                logging.info(f"Finished writing {file_name} to HDFS.")
            except Exception as e:
                logging.error(f"Error uploading file {file_name} to HDFS: {e}")
                raise RuntimeError(f"Failed to upload file {file_name} to HDFS.") from e

        logging.info("Starting upload of files to HDFS.")

        # Collect all file names and data
        all_files = {name: extracted_data.read(name) for name in extracted_data.namelist()}

        # Get list of files already in HDFS
        hdfs_files = client.list(hdfs_path)
        hdfs_files_set = set(hdfs_files)

        # Filter files that are not in HDFS
        files_to_upload = {name: data for name, data in all_files.items() if name not in hdfs_files_set}

        if not files_to_upload:
            logging.info("All files are already present in HDFS.")
        else:
            # Increase the number of threads in ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(upload_file, name, data) for name, data in files_to_upload.items()]
                with tqdm(total=len(futures)) as pbar:
                    for _ in concurrent.futures.as_completed(futures):
                        pbar.update(1)
            logging.info("Finished uploading files to HDFS.")

        # Upload people.csv directly to HDFS
        logging.info("Downloading and uploading people.csv to HDFS.")
        people_response = requests.get('https://cricsheet.org/register/people.csv')
        people_response.raise_for_status()
        people_hdfs_path = os.path.join(hdfs_path, '..', 'people.csv')
        try:
            with client.write(people_hdfs_path, overwrite=True) as writer:
                writer.write(people_response.content)
            logging.info("Successfully uploaded people.csv to HDFS.")
        except Exception as e:
            logging.error(f"Error uploading people.csv to HDFS: {e}")
            raise RuntimeError("Failed to upload people.csv to HDFS.") from e
    except Exception as e:
        logging.error(f"Error uploading files to HDFS: {e}")
        raise RuntimeError("Failed to upload files to HDFS.") from e

    return

def main():
    download_cricsheet()

if __name__ == "__main__":
    main()