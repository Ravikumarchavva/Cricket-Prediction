from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from datapipeline.playerStats.data_loader import load_data, load_players_data
from datapipeline.playerStats.data_preprocessor import preprocess_batting_data, preprocess_bowling_data, preprocess_fielding_data
from datapipeline.playerStats.data_joiner import join_data
from datapipeline.playerStats.data_saver import save_data
from datapipeline.playerStats.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, COUNTRY_CODES
from datapipeline.playerStats.main import create_spark_session, map_country_codes
import os

def preprocess_batting():
    spark = create_spark_session()
    batting_data = load_data(spark, RAW_DATA_DIR, 't20_batting_stats.csv')
    batting_data = preprocess_batting_data(batting_data)
    batting_data = map_country_codes(batting_data, COUNTRY_CODES)
    batting_data.write.save(os.path.join(PROCESSED_DATA_DIR, 'batting_processed.csv'))

def preprocess_bowling():
    spark = create_spark_session()
    bowling_data = load_data(spark, RAW_DATA_DIR, 't20_bowling_stats.csv')
    bowling_data = preprocess_bowling_data(bowling_data)
    bowling_data = map_country_codes(bowling_data, COUNTRY_CODES)
    bowling_data.write.save(os.path.join(PROCESSED_DATA_DIR, 'bowling_processed.csv'))

def preprocess_fielding():
    spark = create_spark_session()
    fielding_data = load_data(spark, RAW_DATA_DIR, 't20_fielding_stats.csv')
    fielding_data = preprocess_fielding_data(fielding_data)
    fielding_data = map_country_codes(fielding_data, COUNTRY_CODES)
    fielding_data.write.save(os.path.join(PROCESSED_DATA_DIR, 'fielding_processed.csv'))

def combine_and_save():
    spark = create_spark_session()
    batting_data = spark.read.csv(os.path.join(PROCESSED_DATA_DIR, 'batting_processed.csv'), header=True, inferSchema=True)
    bowling_data = spark.read.csv(os.path.join(PROCESSED_DATA_DIR, 'bowling_processed.csv'), header=True, inferSchema=True)
    fielding_data = spark.read.csv(os.path.join(PROCESSED_DATA_DIR, 'fielding_processed.csv'), header=True, inferSchema=True)
    players_data = load_players_data(spark, PROCESSED_DATA_DIR)
    player_data = join_data(batting_data, bowling_data, fielding_data)
    save_data(player_data, PROCESSED_DATA_DIR, 'playerstats.csv')

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 1),
    'retries': 1,
}

with DAG('player_stats_pipeline',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dag:

    task_preprocess_batting = PythonOperator(
        task_id='preprocess_batting',
        python_callable=preprocess_batting
    )

    task_preprocess_bowling = PythonOperator(
        task_id='preprocess_bowling',
        python_callable=preprocess_bowling
    )

    task_preprocess_fielding = PythonOperator(
        task_id='preprocess_fielding',
        python_callable=preprocess_fielding
    )

    task_combine_and_save = PythonOperator(
        task_id='combine_and_save',
        python_callable=combine_and_save
    )

    [task_preprocess_batting, task_preprocess_bowling, task_preprocess_fielding] >> task_combine_and_save