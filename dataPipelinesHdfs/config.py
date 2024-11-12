"""
Configuration settings for the T20 Internationals data pipeline.

Contains HDFS, data path, and Spark configuration settings.
"""

import os

# HDFS Configuration
HDFS_HOST = os.getenv('HDFS_HOST', '192.168.245.142')
HDFS_PORT = os.getenv('HDF_PORT', '9870')
HDFS_USER = os.getenv('HDFS_USER', 'ravikumar')
HDFS_URI = f'http://{HDFS_HOST}:{HDFS_PORT}'
NAMENODE_PORT = os.getenv('NAMENODE_PORT', '8020')
HDFS_NAMENODE = f'hdfs://{HDFS_HOST}:{NAMENODE_PORT}'

# HDFS Data Paths
HDFS_BASE_DIR = os.getenv('HDFS_BASE_DIR', '/usr/ravi/t20/data')
RAW_DATA_DIR = os.path.join(HDFS_BASE_DIR, '1_rawData')
PROCESSED_DATA_DIR = os.path.join(HDFS_BASE_DIR, '2_processedData')
MERGED_DATA_DIR = os.path.join(HDFS_BASE_DIR, '3_mergedData')
FILTERED_DATA_DIR = os.path.join(HDFS_BASE_DIR, '4_filteredData')

# Spark Configuration
SPARK_APP_NAME = os.getenv('SPARK_APP_NAME', 'T20Internationals')
SPARK_MASTER_PORT = os.getenv('SPARK_MASTER_PORT', '7077')
SPARK_MASTER = f'spark://{HDFS_HOST}:{SPARK_MASTER_PORT}'
SPARK_CONFIG = {
    'spark.executor.memory': os.getenv('SPARK_EXECUTOR_MEMORY', '4g'),
    'spark.executor.cores': os.getenv('SPARK_EXECUTOR_CORES', '2'),
    'spark.cores.max': os.getenv('SPARK_CORES_MAX', '6'),
    'spark.master': f'spark://{HDFS_HOST}:{SPARK_MASTER_PORT}',
    'spark.hadoop.fs.defaultFS': HDFS_NAMENODE,
    'spark.hadoop.fs.hdfs.impl': 'org.apache.hadoop.hdfs.DistributedFileSystem',
    'spark.hadoop.fs.file.impl': 'org.apache.hadoop.fs.LocalFileSystem'
}
