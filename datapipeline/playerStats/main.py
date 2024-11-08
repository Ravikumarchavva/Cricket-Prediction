
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

def main():
    spark = None
    try:
        logging.info("Starting main process.")
        spark = create_spark_session()

        # Load and preprocess batting data
        batting_data = load_data(spark, RAW_DATA_DIR, 't20_batting_stats.csv')
        batting_data = preprocess_batting_data(batting_data)
        batting_data = map_country_codes(batting_data, COUNTRY_CODES)
        
        # Load and preprocess bowling data
        bowling_data = load_data(spark, RAW_DATA_DIR, 't20_bowling_stats.csv')
        bowling_data = preprocess_bowling_data(bowling_data)
        bowling_data = map_country_codes(bowling_data, COUNTRY_CODES)
        
        # Load and preprocess fielding data
        fielding_data = load_data(spark, RAW_DATA_DIR, 't20_fielding_stats.csv')
        fielding_data = preprocess_fielding_data(fielding_data)
        fielding_data = map_country_codes(fielding_data, COUNTRY_CODES)

        # Load players data
        players_data = load_players_data(spark, PROCESSED_DATA_DIR)

        # Join data with players data
        batting_data = batting_data.join(players_data, ['Player', 'Country'], 'inner')
        bowling_data = bowling_data.join(players_data, ['Player', 'Country'], 'inner')
        fielding_data = fielding_data.join(players_data, ['Player', 'Country'], 'inner')

        # Join all datasets
        player_data = join_data(batting_data, bowling_data, fielding_data)

        # Save the combined data to HDFS
        save_data(player_data, PROCESSED_DATA_DIR, 'playerstats.csv')

        logging.info("Main process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if spark:
            spark.stop()
            logging.info("Spark session stopped.")

def map_country_codes(df, country_codes):
    """Map country codes to full country names and filter data."""
    logging.info("Mapping country codes to full country names.")
    df = df.filter(col('Country').isin(list(country_codes.keys())))
    df = df.replace(country_codes, subset=['Country'])
    return df

if __name__ == "__main__":
    main()
