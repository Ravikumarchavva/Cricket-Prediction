# Configuration settings for the T20 Internationals data pipeline.

# Host Configuration
HOST = 'hdfs-namenode'  # Docker service name for HDFS NameNode

# HDFS Configuration
HDFS_PORT = '9870'  # Port for HDFS Web UI
NAMENODE_PORT = '8020'  # Port for HDFS NameNode
HDFS_USER = 'ravikumar'  # HDFS user for authentication
HDFS_URL = "http://hdfs-namenode:9870"
HDFS_NAMENODE = f"hdfs://{HOST}:{NAMENODE_PORT}"  # HDFS core-site.xml default filesystem URL

# HDFS Data Paths
HDFS_BASE_DIR = '/t20'  # Base directory for T20 pipeline data
RAW_DATA_DIR = f"{HDFS_BASE_DIR}/1_rawData"  # Directory for raw input data
PROCESSED_DATA_DIR = f"{HDFS_BASE_DIR}/2_processedData"  # Directory for processed data
MERGED_DATA_DIR = f"{HDFS_BASE_DIR}/3_mergedData"  # Directory for merged data
FILTERED_DATA_DIR = f"{HDFS_BASE_DIR}/4_filteredData"  # Directory for filtered data

# Spark Configuration
SPARK_MASTER = 'spark-master'  # Docker service name for Spark Master
SPARK_MASTER_PORT = '7077'  # Port for Spark Master
SPARK_MASTER_URL = f"spark://{SPARK_MASTER}:{SPARK_MASTER_PORT}"  # Spark Master URL
SPARK_APP_NAME = 'T20Internationals'  # Spark application name
# Spark Configuration Dictionary
SPARK_CONFIG = {
    'spark.app.name': SPARK_APP_NAME,
    'spark.master': SPARK_MASTER_URL,  # Ensure this is included
    'spark.executor.memory': '2g',
    'spark.executor.cores': '3',
    'spark.hadoop.fs.defaultFS': HDFS_NAMENODE,  # Ensure HDFS is set as default FS
    'spark.dynamicAllocation.enabled': 'true',  # Enable dynamic allocation of executors
    'spark.dynamicAllocation.minExecutors': '0',  # Minimum number of executors
    'spark.dynamicAllocation.maxExecutors': '4',  # Maximum number of executors
    'spark.shuffle.service.enabled': 'false',  # Disable external shuffle service
    'spark.dynamicAllocation.shuffleTracking.enabled': 'true',  # Enable shuffle tracking
    'spark.hadoop.fs.hdfs.impl': 'org.apache.hadoop.hdfs.DistributedFileSystem',  # HDFS implementation class
    'spark.hadoop.fs.file.impl': 'org.apache.hadoop.fs.LocalFileSystem',  # Local filesystem implementation class
}
