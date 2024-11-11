"""Airflow DAG definition for T20 cricket data processing pipeline."""

import sys
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

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

# Import merging tasks
from c_data_merging.tasks import (
    merge_matches_and_deliveries,
    merge_match_team_stats,
    merge_match_players_stats
)

# Import filtering tasks
from d_data_filtering.tasks import filter_data
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

    # Merging tasks
    merge_matches_and_deliveries_task = PythonOperator(
        task_id='merge_matches_and_deliveries',
        python_callable=merge_matches_and_deliveries,
    )

    merge_match_team_stats_task = PythonOperator(
        task_id='merge_match_team_stats',
        python_callable=merge_match_team_stats,
    )

    merge_match_players_stats_task = PythonOperator(
        task_id='merge_match_players_stats',
        python_callable=merge_match_players_stats,
    )

    # Filtering tasks
    filter_data_task = PythonOperator(
        task_id='filter_data',
        python_callable=filter_data,
    )

    # Set task dependencies

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
