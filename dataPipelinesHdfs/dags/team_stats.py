
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Set the PYTHONPATH environment variable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from team_stats.tasks import preprocess_team_stats

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
    'team_stats_pipeline',
    default_args=default_args,
    description='A DAG to process team stats data',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
)

preprocess_team_stats_task = PythonOperator(
    task_id='preprocess_team_stats',
    python_callable=preprocess_team_stats,
    dag=dag,
)