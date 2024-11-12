"""Utility functions for Spark session management and data operations."""

import config
import logging
import os
from pyspark.sql import SparkSession
from hdfs import InsecureClient
import io


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


def spark_save_data(df, output_dir, filename):
    """Save DataFrame to HDFS using Spark's native CSV writer."""
    try:
        output_path = os.path.join(output_dir, filename)
        
        # Use Spark's native CSV writer
        df.write.mode('overwrite').csv(
            output_path,
            header=True
        )
        logging.info(f'Successfully wrote data to {output_path}')
        
    except Exception as e:
        logging.error(f'Error writing to {os.path.join(output_dir, filename)} on HDFS: {str(e)}')
        raise


def get_hdfs_client():
    """Initialize and return an HDFS client."""
    return InsecureClient(url=config.HDFS_URI, user=config.HDFS_USER)

def hdfs_read(client, path):
    """Read a file from HDFS."""
    return client.read(path)

import pandas as pd
import polars as pl

def hdfs_write(client, path, data=None, overwrite=True, encoding='utf-8'):
    """Write data to HDFS (supports Pandas/Polars DataFrames or CSV text data)."""
    
    # Convert DataFrame to CSV if necessary
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False, encoding=encoding)
    elif isinstance(data, pl.DataFrame):
        data = data.write_csv(None)  # Polars writes directly to a string if path is None
    elif isinstance(data, bytes):
        data = data.decode(encoding)
    elif isinstance(data, io.BytesIO):
        data = data.getvalue().decode(encoding)
    elif not isinstance(data, str):
        raise TypeError("Unsupported data type. Provide a Pandas/Polars DataFrame or CSV text data.")

    # Handle existing file or directory if overwrite is True
    if overwrite and hdfs_exists(client, path):
        try:
            client.delete(path, recursive=True)
            logging.info(f"Deleted existing path: {path}")
        except Exception as e:
            logging.error(f"Error deleting {path}: {e}")
            raise

    # Write data to HDFS
    try:
        writer = client.write(path, data=data, overwrite=overwrite, encoding=encoding)
        logging.info(f"Successfully wrote data to {path}")
        return writer
    except Exception as e:
        logging.error(f"Error writing to {path} on HDFS: {e}")
        raise

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


def is_conda_env():
    """Check if the script is running inside a Conda environment."""
    return os.getenv('CONDA_DEFAULT_ENV') is not None


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
