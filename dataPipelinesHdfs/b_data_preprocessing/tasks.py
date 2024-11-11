"""Task definitions for data preprocessing operations."""

import logging

# Import preprocessing functions using relative imports
from b_data_preprocessing.matches import (
    preprocess_matches as process_matches_function
)
from b_data_preprocessing.deliveries import (
    preprocess_deliveries as process_deliveries_function
)
from b_data_preprocessing.people import (
    process_players_data as process_players_function
)
from b_data_preprocessing.team_stats import (
    preprocess_team_data as team_stats_function
)
from b_data_preprocessing.player_stats import (
    preprocess_batting as batting_stats_function,
    preprocess_bowling as bowling_stats_function,
    preprocess_fielding as fielding_stats_function,
    combine_data as combine_stats_function
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


def preprocess_deliveries():
    """Task wrapper for deliveries data processing."""
    logging.info("Starting deliveries data processing task")
    try:
        process_deliveries_function()
        logging.info("Deliveries data processing completed successfully")
    except Exception as e:
        logging.error(f"Error in deliveries data processing task: {e}")
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


def preprocess_batting():
    """Task wrapper for batting stats data processing."""
    logging.info("Starting batting stats data processing task")
    try:
        batting_stats_function()
        logging.info("Batting stats data processing completed successfully")
    except Exception as e:
        logging.error(f"Error in batting stats data processing task: {e}")
        raise


def preprocess_bowling():
    """Task wrapper for bowling stats data processing."""
    logging.info("Starting bowling stats data processing task")
    try:
        bowling_stats_function()
        logging.info("Bowling stats data processing completed successfully")
    except Exception as e:
        logging.error(f"Error in bowling stats data processing task: {e}")
        raise


def preprocess_fielding():
    """Task wrapper for fielding stats data processing."""
    logging.info("Starting fielding stats data processing task")
    try:
        fielding_stats_function()
        logging.info("Fielding stats data processing completed successfully")
    except Exception as e:
        logging.error(f"Error in fielding stats data processing task: {e}")
        raise


def combine_data():
    """Task wrapper for combining player stats data."""
    logging.info("Starting player stats data combining task")
    try:
        combine_stats_function()
        logging.info("Player stats data combining completed successfully")
    except Exception as e:
        logging.error(f"Error in player stats data combining task: {e}")
        raise
