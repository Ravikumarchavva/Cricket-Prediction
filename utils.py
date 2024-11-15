"""Utility functions for Spark session management and data operations."""

import config
import logging
import os
from pyspark.sql import SparkSession
from airflow.providers.apache.hdfs.hooks.webhdfs import WebHDFSHook


def create_spark_session(name : str =None, SPARK_CONFIG : dict = config.SPARK_CONFIG, update_config = False):
    """Create and return a Spark session."""
    logging.info("Creating Spark session.")
    try:
        if not name:
            name = config.SPARK_APP_NAME
        if update_config:
            SPARK_CONFIG.update(config.SPARK_CONFIG)
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
    """Initialize and return an HDFS client using Airflow's HDFSHook."""
    hook = WebHDFSHook(webhdfs_conn_id='webhdfs_default')  # Ensure this matches the connection ID in Airflow
    return hook.get_conn()

def hdfs_read(client, path):
    """Read a file from HDFS."""
    with client.read(path) as reader:
        return reader.read().decode('utf-8')

def hdfs_list(client, path):
    """List files in a directory on HDFS."""
    try:
        return set(client.list(path))
    except Exception as e:
        logging.warning(f"Could not list files in HDFS directory {path}: {e}")
        return set()

def hdfs_mkdirs(client, path):
    """Create directories on HDFS."""
    client.makedirs(path, permission=0o755)

def hdfs_exists(client, path):
    """Check if a path exists on HDFS."""
    try:
        client.status(path)
        return True
    except FileNotFoundError:
        return False

def ensure_hdfs_directory(client, path):
    """Ensure that a directory exists on HDFS, creating it if necessary."""
    if not hdfs_exists(client, path):
        hdfs_mkdirs(client, path)
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