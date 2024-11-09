from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from datetime import datetime, timedelta
import os
import sys

# Set the PYTHONPATH environment variable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from player_stats.tasks import preprocess_batting, preprocess_bowling, preprocess_fielding, combine_data

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
    'player_stats_pipeline',
    default_args=default_args,
    description='A DAG to process player stats data',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
)

wait_for_scrape_espn_stats = ExternalTaskSensor(
    task_id='wait_for_scrape_espn_stats',
    external_dag_id='source_dag',
    external_task_id='scrape_espn_stats',
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
wait_for_scrape_espn_stats >> [preprocess_batting_task, preprocess_bowling_task, preprocess_fielding_task] >> combine_data_task
