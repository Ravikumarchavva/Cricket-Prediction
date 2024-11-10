from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Set the PYTHONPATH environment variable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions from the scripts
from a_data_sources.cricksheet import download_cricsheet
from a_data_sources.scrapping_esp import scrape_espn_stats
from b_data_preprocessing.stats.player_stats.tasks import preprocess_batting, preprocess_bowling, preprocess_fielding, combine_data
from b_data_preprocessing.stats.team_stats.tasks import process_players_data, process_deliveries_data, process_matches_data

default_args = {
    'owner': 'ravikumar',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 10),
}   

dag = DAG(
    't20_dag',
    default_args=default_args,
    description='A DAG to download, scrape, and process player stats data',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
)

download_cricsheet_task = PythonOperator(
    task_id='download_cricsheet',
    python_callable=download_cricsheet,
    dag=dag,
)

scrape_espn_stats_task = PythonOperator(
    task_id='scrape_espn_stats',
    python_callable=scrape_espn_stats,
    dag=dag,
)

process_deliveries_task = PythonOperator(
    task_id='process_deliveries',
    python_callable=process_deliveries_data,
    dag=dag,
)

process_matches_task = PythonOperator(
    task_id='process_matches',
    python_callable=process_matches_data,
    dag=dag,
)

process_players_task = PythonOperator(
    task_id='process_players',
    python_callable=process_players_data,
    dag=dag,
)

preprocess_batting_task = PythonOperator(
    task_id='preprocess_batting',
    python_callable=preprocess_batting,
    dag=dag,
)

preprocess_bowling_task = PythonOperator(
    task_id='preprocess_bowling',
    python_callable=preprocess_bowling,
    dag=dag,
)

preprocess_fielding_task = PythonOperator(
    task_id='preprocess_fielding',
    python_callable=preprocess_fielding,
    dag=dag,
)

combine_data_task = PythonOperator(
    task_id='combine_data',
    python_callable=combine_data,
    dag=dag,
)

# Set task dependencies
download_cricsheet_task >> [process_deliveries_task, process_matches_task]
scrape_espn_stats_task >> [preprocess_batting_task, preprocess_bowling_task, preprocess_fielding_task, process_players_task] >> combine_data_task
[process_matches_task, scrape_espn_stats_task] >> process_players_task