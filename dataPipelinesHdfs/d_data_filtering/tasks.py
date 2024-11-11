from d_data_filtering.filteringData import filter_data
import logging

def filter_data_task():
    """Task wrapper for filtering data."""
    logging.info("Starting data filtering task")
    try:
        filter_data()
        logging.info("Data filtering completed successfully")
    except Exception as e:
        logging.error(f"Error in data filtering task: {e}")
        raise