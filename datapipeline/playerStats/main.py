import logging
from pyspark.sql import SparkSession
from config import BASE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, COUNTRY_CODES
from data_loader import load_data, load_players_data
from data_preprocessor import preprocess_batting_data, preprocess_bowling_data, preprocess_fielding_data
from data_joiner import join_data
from data_saver import save_data
from pyspark.sql.functions import col, regexp_extract, when, row_number, sum as spark_sum

def create_spark_session():
    """Create and return a Spark session."""
    logging.info("Creating Spark session.")
    spark = SparkSession.builder \
        .appName("PlayerStats") \
        .master("spark://192.168.245.142:7077") \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "2") \
        .config("spark.cores.max", "4") \
        .getOrCreate()
    
    # Set Spark logging level to WARN
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

def map_country_codes(df, country_codes):
    """Map country codes to full country names and filter data."""
    logging.info("Mapping country codes to full country names.")
    df = df.filter(col('Country').isin(list(country_codes.keys())))
    df = df.replace(country_codes, subset=['Country'])
    return df

def main():
    pass

if __name__ == "__main__":
    main()
