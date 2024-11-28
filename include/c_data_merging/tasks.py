"""Task definitions for data merging operations."""

import logging
from airflow.decorators import task

# Import merging functions
from c_data_merging.merge_matches_and_deliveries import merge_data
from c_data_merging.merge_match_team_stats import process_match_team_stats
from c_data_merging.merge_match_players_stats import process_match_players_stats

@task
def merge_matches_and_deliveries():
    """Task wrapper for merging matches and deliveries data."""
    logging.info("Starting data merging task")
    try:
        merge_data()
        logging.info("Data merging completed successfully")
    except Exception as e:
        logging.error(f"Error in data merging task: {e}")
        raise

@task
def merge_match_team_stats():
    """Task wrapper for merging match and team stats data."""
    logging.info("Starting match team stats merging task")
    try:
        process_match_team_stats()
        logging.info("Match team stats merging completed successfully")
    except Exception as e:
        logging.error(f"Error in match team stats merging task: {e}")
        raise

@task
def merge_match_players_stats():
    """Task wrapper for merging match players and stats data."""
    logging.info("Starting match players stats merging task")
    try:
        process_match_players_stats()
        logging.info("Match players stats merging completed successfully")
    except Exception as e:
        logging.error(f"Error in match players stats merging task: {e}")
        raise
