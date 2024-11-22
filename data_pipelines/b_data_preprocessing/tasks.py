"""Task definitions for data preprocessing operations."""

import logging

# Import preprocessing functions using relative imports
from b_data_preprocessing.matches import (
    preprocess_matches as process_matches_function
)
from b_data_preprocessing.people import (
    process_players_data as process_players_function
)
from data_pipelines.b_data_preprocessing.player_stats.preprocess_team_data import (
    preprocess_team_data as team_stats_function
)

# define set of tasks to export


def process_players_data():
    """Task wrapper for players data processing."""
    logging.info("Starting players data processing task")
    try:
        process_players_function()
        logging.info("Players data processing completed successfully")
    except Exception as e:
        logging.error(f"Error in players data processing task: {e}")
        raise


def preprocess_matches():
    """Task wrapper for matches data processing."""
    logging.info("Starting matches data processing task")
    try:
        process_matches_function()
        logging.info("Matches data processing completed successfully")
    except Exception as e:
        logging.error(f"Error in matches data processing task: {e}")
        raise



def preprocess_team_data():
    """Task wrapper for team stats data processing."""
    logging.info("Starting team stats data processing task")
    try:
        team_stats_function()
        logging.info("Team stats data processing completed successfully")
    except Exception as e:
        logging.error(f"Error in team stats data processing task: {e}")
        raise