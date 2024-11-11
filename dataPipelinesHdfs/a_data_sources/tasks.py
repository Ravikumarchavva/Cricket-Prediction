"""Task definitions for data source operations."""

import logging
import sys  # Moved to the top
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a_data_sources.scrapping_esp import scrape_and_save_stats as scrape_function
from a_data_sources.cricksheet import download_cricsheet as download_function


def scrape_espn_stats():
    """Task wrapper for ESPN stats scraping."""
    logging.info("Starting ESPN stats scraping task")
    try:
        scrape_function()
        logging.info("ESPN stats scraping completed successfully")
    except Exception as e:
        logging.error(f"Error in ESPN stats scraping task: {e}")
        raise


def download_cricsheet():
    """Task wrapper for Cricsheet data download."""
    logging.info("Starting Cricsheet download task")
    try:
        download_function()
        logging.info("Cricsheet download completed successfully")
    except Exception as e:
        logging.error(f"Error in Cricsheet download task: {e}")
        raise
