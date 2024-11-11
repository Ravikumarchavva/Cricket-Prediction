"""Utility functions for Spark session management and data operations."""

import config
import logging
import os
from pyspark.sql import SparkSession
from hdfs import InsecureClient
import time
import requests
import zipfile
import io
import concurrent.futures
from tqdm import tqdm


def create_spark_session(name=None, SPARK_CONFIG=None):
    """Create and return a Spark session."""
    logging.info("Creating Spark session.")
    try:
        if not name:
            name = config.SPARK_APP_NAME
        if SPARK_CONFIG is None:
            SPARK_CONFIG = config.SPARK_CONFIG
        builder = (SparkSession.builder
                  .appName(name)
                  .master(config.SPARK_MASTER))
        
        # Add all configurations
        for key, value in SPARK_CONFIG.items():
            builder = builder.config(key, value)
        
        spark = builder.getOrCreate()
        
        # Set Spark logging level to WARN
        spark.sparkContext.setLogLevel("WARN")
        logging.info("Spark session created successfully.")
        return spark
    except Exception as e:
        logging.error(f"Error creating Spark session: {e}")
        raise


def load_data(spark, data_dir, filename):
    """Load CSV data."""
    logging.info(f"Loading data from {filename}.")
    return spark.read.csv(
        os.path.join(data_dir, filename),
        header=True,
        inferSchema=True
    )


def save_data(df, hdfs_dir, filename):
    """Save DataFrame to HDFS as CSV."""
    logging.info(f"Saving data to {filename} in HDFS.")
    output_path = os.path.join(hdfs_dir, filename)
    try:
        # Write DataFrame to HDFS as CSV
        df.write.csv(
            output_path,
            header=True,
            mode='overwrite'
        )
        logging.info(f"Data saved successfully to {filename} in HDFS.")
    except Exception as e:
        logging.error(
            f"An error occurred while saving data to {filename} in HDFS: {e}"
        )


def get_hdfs_client():
    """Initialize and return an HDFS client."""
    return InsecureClient(url=config.HDFS_URI, user=config.HDFS_USER)

def hdfs_read(client, path):
    """Read a file from HDFS."""
    return client.read(path)

def hdfs_write(client, path, data=None, overwrite=True, encoding='utf-8', retries=3):
    """Write data to HDFS with retry logic."""
    if isinstance(data, bytes):
        data = data.decode(encoding)
    for attempt in range(retries):
        try:
            return client.write(path, data=data, overwrite=overwrite, encoding=encoding)
        except Exception as e:
            if "Name node is in safe mode" in str(e):
                logging.warning(f"Name node is in safe mode. Retrying write to {path} (attempt {attempt + 1}/{retries})...")
                time.sleep(10)  # Wait for 10 seconds before retrying
            else:
                logging.error(f"Error writing to {path} on HDFS: {e}")
                if attempt < retries - 1:
                    logging.info(f"Retrying write to {path} (attempt {attempt + 1}/{retries})...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed to write to {path} on HDFS after {retries} attempts.") from e

def hdfs_list(client, path):
    """List files in a directory on HDFS."""
    try:
        return set(client.list(path))
    except Exception as e:
        logging.warning(f"Could not list files in HDFS directory {path}: {e}")
        return set()

def hdfs_mkdirs(client, path):
    """Create directories on HDFS."""
    client.makedirs(path)

def hdfs_exists(client, path):
    """Check if a path exists on HDFS."""
    return client.status(path, strict=False) is not None

def ensure_hdfs_directory(client, path):
    """Ensure that a directory exists on HDFS, creating it if necessary."""
    if not hdfs_exists(client, path):
        hdfs_mkdirs(client, path)
        logging.info(f"Created directory: {path}")

def upload_files_to_hdfs(client, hdfs_path, files):
    """Upload multiple files to HDFS."""
    ensure_hdfs_directory(client, hdfs_path)
    hdfs_files_set = hdfs_list(client, hdfs_path)
    files_to_upload = {name: data for name, data in files.items() if name not in hdfs_files_set}

    if not files_to_upload:
        logging.info("All files are already present in HDFS.")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(hdfs_write, client, os.path.join(hdfs_path, name), data) for name, data in files_to_upload.items()]
            with tqdm(total=len(futures)) as pbar:
                for _ in as_completed(futures):
                    pbar.update(1)
        logging.info("Finished uploading files to HDFS.")


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
        client = InsecureClient(config.HDFS_URI, user=config.HDFS_USER)
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
            from concurrent.futures import ThreadPoolExecutor
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


# Country Codes

country_codes = {
    'LES': 'Lesotho', 'BUL': 'Bulgaria', 'VAN': 'Vanuatu', 'ROM': 'Romania',
    'Aut': 'Austria', 'COK': 'Cook Islands', 'Fran': 'France',
    'SRB': 'Serbia', 'PAK': 'Pakistan', 'HUN': 'Hungary', 'CYP': 'Cyprus',
    'Fiji': 'Fiji', 'FIN': 'Finland', 'EST': 'Estonia', 'CHN': 'China',
    'GRC': 'Greece', 'CAM': 'Cambodia', 'GUE': 'Guernsey',
    'SEY': 'Seychelles', 'JPN': 'Japan', 'TAN': 'Tanzania', 'JER': 'Jersey',
    'QAT': 'Qatar', 'ENG': 'England', 'UGA': 'Uganda', 'BER': 'Bermuda',
    'CZK-R': 'Czech Republic', 'CAY': 'Cayman Islands', 'IRE': 'Ireland',
    'Mali': 'Mali', 'BRA': 'Brazil', 'SUI': 'Switzerland', 'Peru': 'Peru',
    'Mex': 'Mexico', 'MOZ': 'Mozambique', 'Samoa': 'Samoa',
    'HKG': 'Hong Kong', 'BAN': 'Bangladesh', 'SL': 'Sri Lanka',
    'PNG': 'Papua New Guinea', 'ZIM': 'Zimbabwe', 'GHA': 'Ghana',
    'SWZ': 'Eswatini', 'MYAN': 'Myanmar', 'IND': 'India',
    'USA': 'United States of America', 'NEP': 'Nepal', 'AFG': 'Afghanistan',
    'PAN': 'Panama', 'NGA': 'Nigeria', 'SLE': 'Sierra Leone', 'ESP': 'Spain',
    'Bhm': 'Bahamas', 'TKY': 'Turkey', 'MWI': 'Malawi', 'WI': 'West Indies',
    'IOM': 'Isle of Man', 'THA': 'Thailand', 'SWA': 'Eswatini', 'SKOR': 'South Korea',
    'GMB': 'Gambia', 'ISR': 'Israel', 'KUW': 'Kuwait', 'Belg': 'Belgium',
    'GER': 'Germany', 'ITA': 'Italy', 'CAN': 'Canada', 'MDV': 'Maldives',
    'Blz': 'Belize', 'DEN': 'Denmark', 'INA': 'Indonesia', 'KENYA': 'Kenya',
    'LUX': 'Luxembourg', 'STHEL': 'Saint Helena', 'BHR': 'Bahrain', 'KSA': 'Saudi Arabia',
    'MLT': 'Malta', 'Arg': 'Argentina', 'MNG': 'Mongolia', 'AUS': 'Australia',
    'GIBR': 'Gibraltar', 'SGP': 'Singapore', 'Chile': 'Chile', 'UAE': 'United Arab Emirates',
    'NZ': 'New Zealand', 'SCOT': 'Scotland', 'BHU': 'Bhutan', 'MAS': 'Malaysia',
    'BOT': 'Botswana', 'CRC': 'Costa Rica', 'PHI': 'Philippines', 'NAM': 'Namibia',
    'RWN': 'Rwanda', 'OMA': 'Oman', 'NOR': 'Norway', 'CRT': 'Croatia',
    'SWE': 'Sweden', 'Iran': 'Iran', 'PORT': 'Portugal', 'NED': 'Netherlands',
    'SA': 'South Africa', 'SVN': 'Slovenia', 'GUE': 'Guernsey', 'MDV': 'Maldives',
    'BHM': 'Bahamas', 'SWE': 'Sweden', 'MLT': 'Malta', 'ITA': 'Italy',
}
