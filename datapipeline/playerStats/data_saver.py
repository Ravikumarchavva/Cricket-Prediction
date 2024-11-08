import os
import logging

def save_data(df, hdfs_dir, filename):
    """Save DataFrame to HDFS as CSV."""
    logging.info(f"Saving data to {filename} in HDFS.")
    output_path = os.path.join(hdfs_dir, filename)
    try:
        # Write DataFrame to HDFS as CSV
        df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')
        logging.info(f"Data saved successfully to {filename} in HDFS.")
    except Exception as e:
        logging.error(f"An error occurred while saving data to {filename} in HDFS: {e}")
