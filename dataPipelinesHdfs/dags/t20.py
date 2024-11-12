"""Airflow DAG definition for T20 cricket data processing pipeline."""

import sys
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.hdfs.hooks.hdfs import HDFSHook
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Sources tasks
from a_data_sources.tasks import download_cricsheet, scrape_espn_stats

# Preprocessing tasks
from b_data_preprocessing.tasks import (
    preprocess_matches,
    process_players_data,
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

    process_deliveries_task = SparkSubmitOperator(
        task_id='preprocess_deliveries',
        application=f'{os.path.join(os.path.dirname(__file__),'..', "b_data_preprocessing", "deliveries.py")}',
        conn_id='spark_default',      
    )

    process_players_task = PythonOperator(
        task_id='process_players_data',
        python_callable=process_players_data,
    )

    preprocess_batting_task = SparkSubmitOperator(
        task_id='preprocess_batting',
        application=f'{os.path.join(os.path.dirname(__file__),'..', "b_data_preprocessing", "preprocess_batting.py")}',
        conn_id='spark_default',      
    )

    preprocess_bowling_task = SparkSubmitOperator(
        task_id='preprocess_bowling',
        application=f'{os.path.join(os.path.dirname(__file__), '..', "b_data_preprocessing", "preprocess_bowling.py")}',
        conn_id='spark_default',
    )

    preprocess_fielding_task = SparkSubmitOperator(
        task_id='preprocess_fielding',
        application=f'{os.path.join(os.path.dirname(__file__), '..' ,"b_data_preprocessing", "preprocess_fielding.py")}',
        conn_id='spark_default',
    )

    preprocess_team_data_task = SparkSubmitOperator(
        task_id='preprocess_team_data',
        application=f'{os.path.join(os.path.dirname(__file__), '..', "b_data_preprocessing", "preprocess_team_data.py")}',
        conn_id='spark_default',
        )

    combine_data_task = SparkSubmitOperator(
        task_id='combine_data',
        application=f'{os.path.join(os.path.dirname(__file__), '..', "b_data_preprocessing", "combine_data.py")}',
        conn_id='spark_default',
        )

    # Merging tasks
    merge_matches_and_deliveries_task = SparkSubmitOperator(
        task_id='merge_matches_and_deliveries',
        application=f'{os.path.join(os.path.dirname(__file__), '..', "c_data_merging", "merge_matches_and_deliveries.py")}',
        conn_id='spark_default',
        )

    merge_match_team_stats_task = SparkSubmitOperator(
        task_id='merge_match_team_stats',
        application=f'{os.path.join(os.path.dirname(__file__), '..',"c_data_merging", "merge_match_team_stats.py")}',
        conn_id='spark_default',
        )

    merge_match_players_stats_task = SparkSubmitOperator(
        task_id='merge_match_players_stats',
        application=f'{os.path.join(os.path.dirname(__file__), '..',"c_data_merging", "merge_match_players_stats.py")}',
        conn_id='spark_default',
        )
    
    # Filtering tasks
    filter_data_task = SparkSubmitOperator(
        task_id='filter_data',
        application=f'{os.path.join(os.path.dirname(__file__), '..',"d_data_filtering", "filter_data.py")}',
        conn_id='spark_default',
        )

    # Define initial tasks
    [download_cricsheet_task, scrape_espn_stats_task]

    # Tasks dependent on download_cricsheet_task
    download_cricsheet_task >> [
        process_deliveries_task,
        process_matches_task,
        process_players_task
    ]

    # Tasks dependent on scrape_espn_stats_task
    scrape_espn_stats_task >> process_players_task

    # Processing tasks dependent on process_players_task
    process_players_task >> [
        preprocess_batting_task,
        preprocess_bowling_task,
        preprocess_fielding_task,
        preprocess_team_data_task
    ]

    # Combine data after all preprocessing tasks are completed
    [
        preprocess_batting_task,
        preprocess_bowling_task,
        preprocess_fielding_task,
        preprocess_team_data_task
    ] >> combine_data_task

    # Set dependencies for combine_data_task
    combine_data_task >> [
        merge_matches_and_deliveries_task,
        merge_match_team_stats_task,
        merge_match_players_stats_task
    ]

    # Ensure process_deliveries_task and process_matches_task are completed before merging
    [process_deliveries_task, process_matches_task] >> merge_matches_and_deliveries_task
    process_matches_task >> merge_match_team_stats_task 
    process_players_task >> merge_match_players_stats_task

    # Set filtering task dependencies
    [
        merge_matches_and_deliveries_task,
        merge_match_team_stats_task,
        merge_match_players_stats_task
    ] >> filter_data_task
