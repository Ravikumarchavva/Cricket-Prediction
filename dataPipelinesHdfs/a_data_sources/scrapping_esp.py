import logging
import os
import sys
import requests
import pandas as pd
import time
from io import StringIO

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import config, utils


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BASE_URL = "https://stats.espncricinfo.com/ci/engine/stats/index.html?class=3;filter=advanced;orderby=season;"

def fetch(session, url, retries=3):
    """Fetch data from a URL using a requests session with retries."""
    for attempt in range(retries):
        try:
            response = session.get(url)
            response.raise_for_status()
            return response.text
        except (requests.RequestException) as e:
            if attempt < retries - 1:
                logging.warning(f"Retrying {url} due to {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logging.error(f"Failed to fetch {url} after {retries} attempts: {e}")
                raise

def scrape_player_stats(session, page):
    """Scrape player stats from ESPN for batting, bowling, and fielding stats on a given page."""
    urls = {
        'batting': (
            f"{BASE_URL}page={page};"
            "size=200;template=results;type=batting;view=season"
        ),
        'bowling': (
            f"{BASE_URL}page={page};"
            "size=200;template=results;type=bowling;view=season"
        ),
        'fielding': (
            f"{BASE_URL}page={page};"
            "size=200;template=results;type=fielding;view=season"
        )
    }
    tasks = {stats_type: fetch(session, url) for stats_type, url in urls.items()}
    responses = [tasks[stats_type] for stats_type in tasks]
    stats_tables = {}

    for stats_type, response in zip(tasks.keys(), responses):
        tables = pd.read_html(StringIO(response), flavor='bs4')
        if len(tables) > 2:
            stats_table = tables[2]
            if len(stats_table) >= 2:
                stats_tables[stats_type] = stats_table
            else:
                logging.info(f"No data on page {page} for {stats_type}. Stopping.")
                break
        else:
            logging.info(f"No table found on page {page} for {stats_type}.")
            break

    return stats_tables

def scrape_team_stats(session):
    """Scrape team stats synchronously."""
    team_tables = []
    page = 1

    while True:
        url = (
            f"{BASE_URL};page={page};size=200;"
            "template=results;type=team;view=season"
        )
        try:
            response = fetch(session, url)
            tables = pd.read_html(StringIO(response), flavor='bs4')
            if len(tables) > 2:
                stats_table = tables[2]
                if len(stats_table) < 2:
                    logging.info(f"No more data for team stats on page {page}. Stopping.")
                    break
                team_tables.append(stats_table)
            else:
                logging.info(f"No table found on page {page} for team stats.")
                break
        except Exception as e:
            logging.error(f"Error fetching team stats on page {page}: {e}")
            break
        page += 1

    if team_tables:
        teams_df = pd.concat(team_tables, ignore_index=True)
        return teams_df
    return pd.DataFrame()

def scrape_and_save_stats():
    """Main function to scrape stats and save them to HDFS."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    with requests.Session() as session:
        session.headers.update(headers)
        # Fetch team stats
        client = utils.get_hdfs_client()
        teams_table = scrape_team_stats(session)
        if not teams_table.empty:
            # Save team stats
            try:
                data = teams_table.to_csv(index=False)
                client.write(f'{config.RAW_DATA_DIR}/t20_team_stats.csv', data=data, overwrite=True)
                logging.info("Successfully saved team stats.")
            except Exception as e:
                logging.error(f"Error writing team stats to HDFS: {e}")
        
        # Fetch player stats (batting, bowling, fielding)
        player_stats = []
        page = 1
        while True:
            stats = scrape_player_stats(session, page)
            if not stats:
                break
            player_stats.append(stats)
            page += 1
            if page % 5 == 0:
                print(f"Page {page} scraped.")
        
        # Process and save each player stat type
        for stats_type, drop_col in [('batting', 'Unnamed: 15'), ('bowling', 'Unnamed: 14'), ('fielding', 'Unnamed: 11')]:
            combined_df = pd.concat([stats[stats_type] for stats in player_stats if stats_type in stats], ignore_index=True)
            if not combined_df.empty:
                combined_df.drop(columns=[drop_col], axis=1, inplace=True, errors='ignore')
                try:
                    data = combined_df.to_csv(index=False)
                    client.write(f'{config.RAW_DATA_DIR}/t20_{stats_type}_stats.csv', data, overwrite=True)
                    logging.info(f"Successfully saved {stats_type} stats.")
                except Exception as e:
                    logging.error(f"Error writing {stats_type} stats to HDFS: {e}")

def main():
    scrape_and_save_stats()

if __name__ == "__main__":
    main()
