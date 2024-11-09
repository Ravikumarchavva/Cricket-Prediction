from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import sys

# Set the PYTHONPATH environment variable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions from the scripts
from datasources.cricksheet import download_cricsheet
from datasources.scrapping_esp import scrape_espn_stats

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'source_dag',
    default_args=default_args,
    description='A DAG to run cricksheet and scrapping_esp in parallel',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
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

[download_cricsheet_task, scrape_espn_stats_task]