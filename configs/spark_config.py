
# Configuration settings for the T20 Internationals data pipeline.
# HOST = '192.168.245.142'
HOST = 'localhost' # docker

# HDFS Configuration
HDFS_PORT = '9870'
NAMENODE_PORT = '8020'
HDFS_USER = 'ravikumar'
HDFS_URL = f"http://{HOST}:{HDFS_PORT}"
HDFS_NAMENODE = f"hdfs://{HOST}:{NAMENODE_PORT}"  # core-site.xml

# HDFS Data Paths
HDFS_BASE_DIR = '/usr/ravi/t20/data'
RAW_DATA_DIR = f"{HDFS_BASE_DIR}/1_rawData"
PROCESSED_DATA_DIR = f"{HDFS_BASE_DIR}/2_processedData"
MERGED_DATA_DIR = f"{HDFS_BASE_DIR}/3_mergedData"
FILTERED_DATA_DIR = f"{HDFS_BASE_DIR}/4_filteredData"

# Spark Configuration
SPARK_APP_NAME = 'T20Internationals'
SPARK_MASTER_PORT = '7077'
SPARK_MASTER = f"spark://{HOST}:{SPARK_MASTER_PORT}"
SPARK_CONFIG = {
    'spark.master': SPARK_MASTER,
    'spark.executor.memory': '2g',
    'spark.executor.cores': '3',
    'spark.dynamicAllocation.enabled': 'true',
    'spark.dynamicAllocation.minExecutors': '0',
    'spark.shuffle.service.enabled': 'true',
    'spark.dynamicAllocation.shuffleTracking.enabled': 'true',
    'spark.dynamicAllocation.maxExecutors': '4',
    'spark.hadoop.fs.defaultFS': HDFS_NAMENODE,
    'spark.hadoop.fs.hdfs.impl': 'org.apache.hadoop.hdfs.DistributedFileSystem',
    'spark.hadoop.fs.file.impl': 'org.apache.hadoop.fs.LocalFileSystem'
}