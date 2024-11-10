import pandas as pd
import requests
import logging
from hdfs import InsecureClient
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def fetch(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()

async def scrape_stats():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        batting_table = pd.DataFrame()
        bowling_table = pd.DataFrame()
        fielding_table = pd.DataFrame()
        i = 1
        while True:
            try:
                if i % 5 == 0:
                    logging.info(f"Processing page {i}")
                urls = {
                    'batting': f"https://stats.espncricinfo.com/ci/engine/stats/index.html?class=3;filter=advanced;orderby=season;page={i};size=200;template=results;type=batting;view=season",
                    'bowling': f"https://stats.espncricinfo.com/ci/engine/stats/index.html?class=3;filter=advanced;orderby=season;page={i};size=200;template=results;type=bowling;view=season",
                    'fielding': f"https://stats.espncricinfo.com/ci/engine/stats/index.html?class=3;filter=advanced;orderby=season;page={i};size=200;template=results;type=fielding;view=season"
                }
                tasks = {stats_type: fetch(session, url) for stats_type, url in urls.items()}
                responses = await asyncio.gather(*tasks.values())
                stop = False
                for stats_type, response in zip(tasks.keys(), responses):
                    tables = pd.read_html(response, flavor='bs4')
                    if len(tables) > 2:
                        stats_table = tables[2]
                        if len(stats_table) < 2:
                            logging.info(f"Table at page {i} for {stats_type} has less than 2 rows. Stopping.")
                            stop = True
                            break
                        if stats_type == 'batting':
                            batting_table = pd.concat([stats_table, batting_table], axis=0, ignore_index=True)
                        elif stats_type == 'bowling':
                            bowling_table = pd.concat([stats_table, bowling_table], axis=0, ignore_index=True)
                        elif stats_type == 'fielding':
                            fielding_table = pd.concat([stats_table, fielding_table], axis=0, ignore_index=True)
                    else:
                        logging.info(f"No more tables found at page {i} for {stats_type}.")
                        stop = True
                        break
                if stop:
                    break
                i += 1
            except Exception as e:
                logging.error(f"Error scraping player stats at page {i}: {e}")
                print(e)
                break
        return batting_table, bowling_table, fielding_table

def scrape_espn_stats():
    """Scrape team, batting, bowling, and fielding stats from ESPN and save to HDFS."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        logging.info("Initializing HDFS client.")
        client = InsecureClient('http://192.168.245.142:9870', user='ravikumar')
        hdfs_path = '/usr/ravi/t20/data/1_rawData'  # Corrected path

        logging.info("Starting scraping of team stats.")
        teams_tables = []
        i = 0
        while True:
            try:
                url = f"https://stats.espncricinfo.com/ci/engine/stats/index.html?class=3;filter=advanced;orderby=season;page={i};size=200;template=results;type=team;view=season"
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                tables = pd.read_html(response.text, flavor='bs4')
                if len(tables) > 2:
                    if len(tables[2]) < 2:
                        logging.info(f"Table at page {i} has less than 2 rows. Stopping.")
                        break
                    stats_table = tables[2]
                    teams_tables.append(stats_table)
                    i += 1
                else:
                    logging.info(f"No more tables found at page {i}.")
                    break
            except Exception as e:
                logging.error(f"Error scraping team stats at page {i}: {e}")
                print(e)
                break
            finally:
                logging.info(f"Finished scraping team stats at page {i}.")
        if teams_tables:
            teams_table = pd.concat(teams_tables, ignore_index=True)
            teams_table.sort_values(by=['Team', 'Season'], ascending=False, inplace=True)
            teams_table.drop(columns=['Unnamed: 13'], axis=1, inplace=True)
            try:
                with client.write(f'{hdfs_path}/t20_team_stats.csv', encoding='utf-8', overwrite=True) as writer:
                    teams_table.to_csv(writer, index=False)
                logging.info("Finished writing team stats.")
            except Exception as e:
                logging.error(f"Error writing team stats to HDFS: {e}")
        else:
            logging.error("No team stats tables were scraped.")

        logging.info("Starting scraping of player stats.")
        batting_table, bowling_table, fielding_table = asyncio.run(scrape_stats())
        drop_columns = {'batting': 'Unnamed: 15', 'bowling': 'Unnamed: 14', 'fielding': 'Unnamed: 11'}
        
        if not batting_table.empty:
            batting_table.sort_values(by=['Player', 'Season'], ascending=False, inplace=True)
            batting_table.drop(columns=[drop_columns['batting']], axis=1, inplace=True)
            try:
                with client.write(f'{hdfs_path}/t20_batting_stats.csv', encoding='utf-8', overwrite=True) as writer:
                    batting_table.to_csv(writer, index=False)
                logging.info("Finished writing batting stats.")
            except Exception as e:
                logging.error(f"Error writing batting stats to HDFS: {e}")
        else:
            logging.error("No batting stats tables were scraped.")

        if not bowling_table.empty:
            bowling_table.sort_values(by=['Player', 'Season'], ascending=False, inplace=True)
            bowling_table.drop(columns=[drop_columns['bowling']], axis=1, inplace=True)
            try:
                with client.write(f'{hdfs_path}/t20_bowling_stats.csv', encoding='utf-8', overwrite=True) as writer:
                    bowling_table.to_csv(writer, index=False)
                logging.info("Finished writing bowling stats.")
            except Exception as e:
                logging.error(f"Error writing bowling stats to HDFS: {e}")
        else:
            logging.error("No bowling stats tables were scraped.")

        if not fielding_table.empty:
            fielding_table.sort_values(by=['Player', 'Season'], ascending=False, inplace=True)
            fielding_table.drop(columns=[drop_columns['fielding']], axis=1, inplace=True)
            try:
                with client.write(f'{hdfs_path}/t20_fielding_stats.csv', encoding='utf-8', overwrite=True) as writer:
                    fielding_table.to_csv(writer, index=False)
                logging.info("Finished writing fielding stats.")
            except Exception as e:
                logging.error(f"Error writing fielding stats to HDFS: {e}")
        else:
            logging.error("No fielding stats tables were scraped.")
    except Exception as e:
        logging.error(f"Error during scraping: {e}")
        print(e)

def main():
    scrape_espn_stats()

if __name__ == "__main__":
    main()




