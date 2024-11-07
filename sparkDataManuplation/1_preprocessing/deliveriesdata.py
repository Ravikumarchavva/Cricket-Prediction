#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob

# Specify the directory where your CSV files are located
directory = r'D:\github\Cricket-Prediction\data\1_rawData\t20s_csv2'

# Use glob to find all CSV files in the specified directory
info_files = glob.glob(os.path.join(directory, '*_info.csv'))
all_files = glob.glob(os.path.join(directory,'*.csv'))
delivery_files = [file for file in all_files if '_info' not in file]

matches=[]
deliveries=[]
# Print the list of CSV files
for info_file in info_files:
    matches.append(info_file.split('\\')[-1])
for delivery in delivery_files:
    if '_info' not in delivery:
        deliveries.append(delivery.split('\\')[-1])


# In[2]:


from pyspark.sql.types import *
import os
from pyspark.sql import SparkSession
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.appName('deliveries').getOrCreate()

# Define the schema for the deliveries data
delivery_schema = StructType([
    StructField('match_id', IntegerType(), True),
    StructField('season', StringType(), True),
    StructField('start_date', StringType(), True),
    StructField('venue', StringType(), True),
    StructField('innings', IntegerType(), True),
    StructField('ball', FloatType(), True),
    StructField('batting_team', StringType(), True),
    StructField('bowling_team', StringType(), True),
    StructField('striker', StringType(), True),
    StructField('non_striker', StringType(), True),
    StructField('bowler', StringType(), True),
    StructField('runs_off_bat', IntegerType(), True),
    StructField('extras', IntegerType(), True),
    StructField('wides', IntegerType(), True),
    StructField('noballs', StringType(), True),
    StructField('byes', IntegerType(), True),
    StructField('legbyes', IntegerType(), True),
    StructField('penalty', StringType(), True),
    StructField('wicket_type', StringType(), True),
    StructField('player_dismissed', StringType(), True),
    StructField('other_wicket_type', StringType(), True),
    StructField('other_player_dismissed', StringType(), True)
])

# Initialize an empty DataFrame with the schema
deliveries_data = spark.read.csv(delivery_files, header=True, schema=delivery_schema)
deliveries_data.show(5)


# In[3]:


from pyspark.sql.functions import col, sum

# Count the number of null values in each column
null_counts = deliveries_data.select([sum(col(c).isNull().cast("int")).alias(c) for c in deliveries_data.columns])
null_counts.show()


# In[4]:


deliveries_data = deliveries_data.fillna(0)
deliveries_data.show(5)


# In[5]:


null_counts = deliveries_data.select([sum(col(c).isNull().cast("int")).alias(c) for c in deliveries_data.columns])
null_counts.show()


# In[6]:


deliveries_data.printSchema()


# In[7]:


from pyspark.sql.functions import when

deliveries_data = deliveries_data.withColumn('noballs', when(col('noballs').isNull(), '0').otherwise(col('noballs')).cast(IntegerType()))
deliveries_data = deliveries_data.withColumn('penalty', when(col('penalty').isNull(), '0').otherwise(col('penalty')).cast(IntegerType()))
deliveries_data.show(5)


# In[8]:


from pyspark.sql.functions import when
columns = ['wicket_type','player_dismissed','other_wicket_type','other_player_dismissed']
for column in columns:
    deliveries_data = deliveries_data.withColumn(column, when(col(column).isNull(), '0').otherwise('1').cast(IntegerType()))

deliveries_data.show()


# In[9]:


# Write the DataFrame to a Parquet file
# deliveries_data.write.mode('overwrite').parquet(os.path.join(r'D:\github\Cricket-Prediction\data\2_processedData','deliveries.parquet'))

# For windows

import pandas as pd
deliveries_data.toPandas().to_parquet(os.path.join(r'D:\github\Cricket-Prediction\data\2_processedData','deliveries.parquet'),index=False)

