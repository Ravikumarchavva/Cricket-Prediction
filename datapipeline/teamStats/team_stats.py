#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os

# Specify the directory where your CSV files are located
directory = r'D:\github\Cricket-Prediction\data\1_rawData' 

# sparksession
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("CricketPrediction").getOrCreate()

team_data = spark.read.csv(os.path.join(directory, 't20_team_stats.csv'), header=True, inferSchema=True)
team_data.show()


# In[8]:


from pyspark.sql.functions import col,when,round
team_data = team_data.withColumn("W/L", round(when(col("Lost")==0, col("Won")).otherwise(col("Won")/col("Lost")),2))
team_data = team_data.withColumn("AveRPW", when(col("Ave")=='-',0).otherwise(col("Ave")).cast("float")).drop("Ave")
team_data = team_data.withColumn("AveRPO", when(col("RPO")=='-',0).otherwise(col("RPO")).cast("float")).drop("RPO","LS")
team_data.show()


# In[9]:


# Cumulative calculations
from pyspark.sql import Window
from pyspark.sql.functions import col, sum as spark_sum, when, row_number, round

# Define the window specification for cumulative calculations
window_spec = Window.partitionBy("Team").orderBy("Season").rowsBetween(Window.unboundedPreceding, -1)

# Window for row number to identify the first row per player and country
row_num_window = Window.partitionBy("Team").orderBy("Season")

# perform cumulative calculations
team_data = team_data.withColumn("row_num", row_number().over(row_num_window)) \
    .withColumn("Cumulative Won",
                when(col("row_num") == 1, 0)
                .otherwise(spark_sum("Won").over(window_spec))) \
    .withColumn("Cumulative Lost",
                when(col("row_num") == 1, 0)  # Set 0 for the first row (before any match)
                .otherwise(spark_sum("Lost").over(window_spec))) \
    .withColumn("Cumulative Tied", 
                when(col("row_num") == 1, 0)  # Set 0 for the first row (before any match)
                .otherwise(spark_sum("Tied").over(window_spec))) \
    .withColumn("Cumulative NR", 
                when(col("row_num") == 1, 0)
                .otherwise(spark_sum("NR").over(window_spec))) \
    .withColumn("Cumulative W/L", 
                when(col("row_num") == 1, 0)
                .otherwise(
                    round(
                        when(spark_sum("Lost").over(window_spec) != 0, 
                             spark_sum(("Won")).over(window_spec) / spark_sum("Lost").over(window_spec))
                        .otherwise(0), 2)
                )
    ) \
    .withColumn("Cumulative AveRPW", 
                when(col("row_num") == 1, 0)
                .otherwise(
                    round(
                        when(spark_sum("Won").over(window_spec) != 0, 
                             spark_sum(col("AveRPW")*col("Mat")).over(window_spec) / spark_sum("Mat").over(window_spec))
                        .otherwise(0), 2)
                )
    ) \
    .withColumn("Cumulative AveRPO", 
                when(col("row_num") == 1, 0)
                .otherwise(
                    round(
                        when(spark_sum("Lost").over(window_spec) != 0, 
                             spark_sum(col("AveRPO")*col("Mat")).over(window_spec) / spark_sum("Mat").over(window_spec))
                        .otherwise(0), 2)
                )
    ) \
    .drop("row_num")  # Drop the temporary row number column

# Show the resulting DataFrame
team_data.show(10)


# In[10]:


team_data = team_data.select("Team", "Season","Cumulative Won", "Cumulative Lost", "Cumulative Tied", "Cumulative NR", "Cumulative W/L", "Cumulative AveRPW", "Cumulative AveRPO")
team_data.show(5)


# In[11]:


team_data.toPandas().to_csv(r'D:\github\Cricket-Prediction\data\2_processedData\teamStats.csv', index=False)


# In[ ]:




