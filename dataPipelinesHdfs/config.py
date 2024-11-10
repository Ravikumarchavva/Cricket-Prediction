import os
import logging
from pyspark.sql import SparkSession

# HDFS Configuration
HDFS_HOST = '192.168.245.142'
HDFS_PORT = '8020'
HDFS_HTTP_PORT = '9870'
HDFS_USER = 'ravikumar'
HDFS_URI = f'hdfs://{HDFS_HOST}:{HDFS_PORT}'
HDFS_BASE_DIR = '/usr/ravi/t20/data'

# HDFS Data Paths
RAW_DATA_DIR = os.path.join(HDFS_BASE_DIR, 'data', '1_rawData')
PROCESSED_DATA_DIR = os.path.join(HDFS_BASE_DIR, 'data', '2_processedData')

# Spark Configuration
SPARK_APP_NAME = 'T20Internationals'
SPARK_MASTER_PORT = '7077'
SPARK_MASTER = F'spark://{HDFS_HOST}:{SPARK_MASTER_PORT}'
SPARK_CONFIG = {
    'spark.executor.memory': '2g',
    'spark.executor.cores': '2',
    'spark.cores.max': '6'
}

def create_spark_session(SPARK_CONFIG : dict = SPARK_CONFIG):
    """Create and return a Spark session."""
    logging.info("Creating Spark session.")
    try:
        builder = SparkSession.builder.appName(SPARK_APP_NAME).master(SPARK_MASTER)
        
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



# Country Codes

country_codes = {
        'LES': 'Lesotho', 'BUL': 'Bulgaria', 'VAN': 'Vanuatu', 'ROM': 'Romania', 'Aut': 'Austria', 'COK': 'Cook Islands', 'Fran': 'France', 'SRB': 'Serbia', 'PAK': 'Pakistan', 'HUN': 'Hungary', 'CYP': 'Cyprus', 'Fiji': 'Fiji', 'FIN': 'Finland', 'EST': 'Estonia', 'CHN': 'China', 'GRC': 'Greece', 'CAM': 'Cambodia', 'GUE': 'Guernsey', 'SEY': 'Seychelles', 'JPN': 'Japan', 'TAN': 'Tanzania', 'JER': 'Jersey', 'QAT': 'Qatar', 'ENG': 'England', 'UGA': 'Uganda', 'BER': 'Bermuda', 'CZK-R': 'Czech Republic', 'CAY': 'Cayman Islands', 'IRE': 'Ireland', 'Mali': 'Mali', 'BRA': 'Brazil', 'SUI': 'Switzerland', 'Peru': 'Peru', 'Mex': 'Mexico', 'MOZ': 'Mozambique', 'Samoa': 'Samoa', 'HKG': 'Hong Kong', 'BAN': 'Bangladesh', 'SL': 'Sri Lanka', 'PNG': 'Papua New Guinea', 'ZIM': 'Zimbabwe', 'GHA': 'Ghana', 'SWZ': 'Eswatini', # Swaziland's official name now is Eswatini
        'MYAN': 'Myanmar', 'IND': 'India', 'USA': 'United States of America', 'NEP': 'Nepal', 'AFG': 'Afghanistan', 'PAN': 'Panama', 'NGA': 'Nigeria', 'SLE': 'Sierra Leone', 'ESP': 'Spain', 'Bhm': 'Bahamas', 'TKY': 'Turkey', 'MWI': 'Malawi', 'WI': 'West Indies', 'IOM': 'Isle of Man', 'THA': 'Thailand', 'SWA': 'Eswatini',
        'SKOR': 'South Korea', 'GMB': 'Gambia', 'ISR': 'Israel', 'KUW': 'Kuwait', 'Belg': 'Belgium', 'GER': 'Germany', 'ITA': 'Italy', 'CAN': 'Canada', 'MDV': 'Maldives', 'Blz': 'Belize', 'DEN': 'Denmark', 'INA': 'Indonesia', 'KENYA': 'Kenya', 'LUX': 'Luxembourg', 'STHEL': 'Saint Helena', 'BHR': 'Bahrain', 'KSA': 'Saudi Arabia', 'MLT': 'Malta', 'Arg': 'Argentina', 'MNG': 'Mongolia', 'AUS': 'Australia', 'GIBR': 'Gibraltar', 'SGP': 'Singapore', 'Chile': 'Chile', 'UAE': 'United Arab Emirates', 'NZ': 'New Zealand', 'SCOT': 'Scotland', 'BHU': 'Bhutan', 'MAS': 'Malaysia', 'BOT': 'Botswana', 'CRC': 'Costa Rica', 'PHI': 'Philippines', 'NAM': 'Namibia', 'RWN': 'Rwanda', 'OMA': 'Oman', 'NOR': 'Norway', 'CRT': 'Croatia', 'SWE': 'Sweden', 'Iran': 'Iran', 'PORT': 'Portugal', 'NED': 'Netherlands', 'SA': 'South Africa', 'SVN': 'Slovenia', 'GUE': 'Guernsey', 'MDV': 'Maldives', 'BHM': 'Bahamas', 'SWE': 'Sweden', 'MLT': 'Malta', 'ITA': 'Italy',
    }
