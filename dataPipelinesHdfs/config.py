"""
Configuration settings for the T20 Internationals data pipeline.

Contains HDFS, data path, and Spark configuration settings.
"""

import os

# HDFS Configuration
HDFS_HOST = '192.168.245.142'
HDFS_PORT = '8020'
HDFS_HTTP_PORT = '9870'
HDFS_USER = 'ravikumar'
HDFS_URI = f'hdfs://{HDFS_HOST}:{HDFS_PORT}'
HDFS_BASE_DIR = '/usr/ravi/t20/data'

# HDFS Data Paths
RAW_DATA_DIR = os.path.join(HDFS_BASE_DIR, '1_rawData')
PROCESSED_DATA_DIR = os.path.join(HDFS_BASE_DIR, '2_processedData')

# Spark Configuration
SPARK_APP_NAME = 'T20Internationals'
SPARK_MASTER_PORT = '7077'
SPARK_MASTER = f'spark://{HDFS_HOST}:{SPARK_MASTER_PORT}'
SPARK_CONFIG = {
    'spark.executor.memory': '2g',
    'spark.executor.cores': '2',
    'spark.cores.max': '6',
    'spark.master': 'spark://192.168.245.142:7077'
}
