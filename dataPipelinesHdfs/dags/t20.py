from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Sources tasks
from a_data_sources.tasks import download_cricsheet, scrape_espn_stats

# Preprocessing tasks
from b_data_preprocessing.tasks import (
    preprocess_matches,
    preprocess_deliveries,
    process_players_data,
    preprocess_batting,
    preprocess_bowling,
    preprocess_fielding,
    preprocess_team_data,
    combine_data
)

default_args = {
    'owner': 'ravikumar',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 10),
}   

with DAG(
    't20_dag',
    default_args=default_args,
    description='A DAG to download, scrape, and process player stats data',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
) as dag:
    
    # Sources
    download_cricsheet_task = PythonOperator(
        task_id='download_cricsheet',
        python_callable=download_cricsheet,
    )
    
    scrape_espn_stats_task = PythonOperator(
        task_id='scrape_espn_stats',
        python_callable=scrape_espn_stats,
    )
    
    # Preprocessing tasks
    process_matches_task = PythonOperator(
        task_id='preprocess_matches',
        python_callable=preprocess_matches,
    )
    
    process_deliveries_task = PythonOperator(
        task_id='preprocess_deliveries',
        python_callable=preprocess_deliveries,
    )

    process_players_task = PythonOperator(
        task_id='process_players_data',
        python_callable=process_players_data,
    )
    
    preprocess_batting_task = PythonOperator(
        task_id='preprocess_batting',
        python_callable=preprocess_batting,
    )
    
    preprocess_bowling_task = PythonOperator(
        task_id='preprocess_bowling',
        python_callable=preprocess_bowling,
    )
    
    preprocess_fielding_task = PythonOperator(
        task_id='preprocess_fielding',
        python_callable=preprocess_fielding,
    )
    
    preprocess_team_data_task = PythonOperator(
        task_id='preprocess_team_data',
        python_callable=preprocess_team_data,
    )

    combine_data_task = PythonOperator(
        task_id='combine_data',
        python_callable=combine_data,
    )

    # Set task dependencies
    [download_cricsheet_task, scrape_espn_stats_task]
    
    download_cricsheet_task >> [process_deliveries_task, process_matches_task, process_players_task]
    
    scrape_espn_stats_task >> process_players_task >> [
        preprocess_batting_task,
        preprocess_bowling_task,
        preprocess_fielding_task,
        preprocess_team_data_task
    ] >> combine_data_task
