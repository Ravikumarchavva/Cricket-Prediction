# Databricks notebook source
# Access secrets from the Key Vault
storage_account_secret = dbutils.secrets.get(scope="key-vault-scope", key="twentytwentydata")

# Define the storage account and container details
storage_account_name = 'twentytwentydata'
container_name = "sink"

# Set up the configuration for accessing the storage account
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_secret)

# COMMAND ----------

# Read data from the specified container
data_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/output.csv"
df = spark.read.format("csv").option("header", "true").load(data_path)

# Display the data
display(df)

# COMMAND ----------

data = df.select(['match_id','innings','ball','batting_team','bowling_team','venue','winner'])
display(data)

# COMMAND ----------


