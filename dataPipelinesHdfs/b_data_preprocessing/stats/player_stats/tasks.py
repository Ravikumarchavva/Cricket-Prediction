
import os
import logging
from b_data_preprocessing.stats.player_stats.utils import create_spark_session, load_data, save_data
from b_data_preprocessing.stats.player_stats.preprocessing import preprocess_batting_data, preprocess_bowling_data, preprocess_fielding_data, map_country_codes



def preprocess_batting():
    logging.info("Starting preprocess_batting task.")
    spark = create_spark_session()
    base_dir = "hdfs://192.168.245.142:8020/usr/ravi/t20"
    raw_data_dir = os.path.join(base_dir, 'data', '1_rawData')
    processed_data_dir = os.path.join(base_dir, 'data', '2_processedData')

    country_codes = {
            'LES': 'Lesotho', 'BUL': 'Bulgaria', 'VAN': 'Vanuatu', 'ROM': 'Romania', 'Aut': 'Austria', 'COK': 'Cook Islands', 'Fran': 'France', 'SRB': 'Serbia', 'PAK': 'Pakistan', 'HUN': 'Hungary', 'CYP': 'Cyprus', 'Fiji': 'Fiji', 'FIN': 'Finland', 'EST': 'Estonia', 'CHN': 'China', 'GRC': 'Greece', 'CAM': 'Cambodia', 'GUE': 'Guernsey', 'SEY': 'Seychelles', 'JPN': 'Japan', 'TAN': 'Tanzania', 'JER': 'Jersey', 'QAT': 'Qatar', 'ENG': 'England', 'UGA': 'Uganda', 'BER': 'Bermuda', 'CZK-R': 'Czech Republic', 'CAY': 'Cayman Islands', 'IRE': 'Ireland', 'Mali': 'Mali', 'BRA': 'Brazil', 'SUI': 'Switzerland', 'Peru': 'Peru', 'Mex': 'Mexico', 'MOZ': 'Mozambique', 'Samoa': 'Samoa', 'HKG': 'Hong Kong', 'BAN': 'Bangladesh', 'SL': 'Sri Lanka', 'PNG': 'Papua New Guinea', 'ZIM': 'Zimbabwe', 'GHA': 'Ghana', 'SWZ': 'Eswatini', # Swaziland's official name now is Eswatini
            'MYAN': 'Myanmar', 'IND': 'India', 'USA': 'United States of America', 'NEP': 'Nepal', 'AFG': 'Afghanistan', 'PAN': 'Panama', 'NGA': 'Nigeria', 'SLE': 'Sierra Leone', 'ESP': 'Spain', 'Bhm': 'Bahamas', 'TKY': 'Turkey', 'MWI': 'Malawi', 'WI': 'West Indies', 'IOM': 'Isle of Man', 'THA': 'Thailand', 'SWA': 'Eswatini',
            'SKOR': 'South Korea', 'GMB': 'Gambia', 'ISR': 'Israel', 'KUW': 'Kuwait', 'Belg': 'Belgium', 'GER': 'Germany', 'ITA': 'Italy', 'CAN': 'Canada', 'MDV': 'Maldives', 'Blz': 'Belize', 'DEN': 'Denmark', 'INA': 'Indonesia', 'KENYA': 'Kenya', 'LUX': 'Luxembourg', 'STHEL': 'Saint Helena', 'BHR': 'Bahrain', 'KSA': 'Saudi Arabia', 'MLT': 'Malta', 'Arg': 'Argentina', 'MNG': 'Mongolia', 'AUS': 'Australia', 'GIBR': 'Gibraltar', 'SGP': 'Singapore', 'Chile': 'Chile', 'UAE': 'United Arab Emirates', 'NZ': 'New Zealand', 'SCOT': 'Scotland', 'BHU': 'Bhutan', 'MAS': 'Malaysia', 'BOT': 'Botswana', 'CRC': 'Costa Rica', 'PHI': 'Philippines', 'NAM': 'Namibia', 'RWN': 'Rwanda', 'OMA': 'Oman', 'NOR': 'Norway', 'CRT': 'Croatia', 'SWE': 'Sweden', 'Iran': 'Iran', 'PORT': 'Portugal', 'NED': 'Netherlands', 'SA': 'South Africa', 'SVN': 'Slovenia', 'GUE': 'Guernsey', 'MDV': 'Maldives', 'BHM': 'Bahamas', 'SWE': 'Sweden', 'MLT': 'Malta', 'ITA': 'Italy',
        }

    try:
        batting_data = load_data(spark, raw_data_dir, 't20_batting_stats.csv')
        batting_data = preprocess_batting_data(batting_data)
        batting_data = map_country_codes(batting_data, country_codes)
        save_data(batting_data, processed_data_dir, 'batting_data.csv')
        logging.info("Batting data processing and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in preprocess_batting task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

def preprocess_bowling():
    logging.info("Starting preprocess_bowling task.")
    spark = create_spark_session()
    base_dir = "hdfs://192.168.245.142:8020/usr/ravi/t20"
    raw_data_dir = os.path.join(base_dir, 'data', '1_rawData')
    processed_data_dir = os.path.join(base_dir, 'data', '2_processedData')

    country_codes = {
            'LES': 'Lesotho', 'BUL': 'Bulgaria', 'VAN': 'Vanuatu', 'ROM': 'Romania', 'Aut': 'Austria', 'COK': 'Cook Islands', 'Fran': 'France', 'SRB': 'Serbia', 'PAK': 'Pakistan', 'HUN': 'Hungary', 'CYP': 'Cyprus', 'Fiji': 'Fiji', 'FIN': 'Finland', 'EST': 'Estonia', 'CHN': 'China', 'GRC': 'Greece', 'CAM': 'Cambodia', 'GUE': 'Guernsey', 'SEY': 'Seychelles', 'JPN': 'Japan', 'TAN': 'Tanzania', 'JER': 'Jersey', 'QAT': 'Qatar', 'ENG': 'England', 'UGA': 'Uganda', 'BER': 'Bermuda', 'CZK-R': 'Czech Republic', 'CAY': 'Cayman Islands', 'IRE': 'Ireland', 'Mali': 'Mali', 'BRA': 'Brazil', 'SUI': 'Switzerland', 'Peru': 'Peru', 'Mex': 'Mexico', 'MOZ': 'Mozambique', 'Samoa': 'Samoa', 'HKG': 'Hong Kong', 'BAN': 'Bangladesh', 'SL': 'Sri Lanka', 'PNG': 'Papua New Guinea', 'ZIM': 'Zimbabwe', 'GHA': 'Ghana', 'SWZ': 'Eswatini', # Swaziland's official name now is Eswatini
            'MYAN': 'Myanmar', 'IND': 'India', 'USA': 'United States of America', 'NEP': 'Nepal', 'AFG': 'Afghanistan', 'PAN': 'Panama', 'NGA': 'Nigeria', 'SLE': 'Sierra Leone', 'ESP': 'Spain', 'Bhm': 'Bahamas', 'TKY': 'Turkey', 'MWI': 'Malawi', 'WI': 'West Indies', 'IOM': 'Isle of Man', 'THA': 'Thailand', 'SWA': 'Eswatini',
            'SKOR': 'South Korea', 'GMB': 'Gambia', 'ISR': 'Israel', 'KUW': 'Kuwait', 'Belg': 'Belgium', 'GER': 'Germany', 'ITA': 'Italy', 'CAN': 'Canada', 'MDV': 'Maldives', 'Blz': 'Belize', 'DEN': 'Denmark', 'INA': 'Indonesia', 'KENYA': 'Kenya', 'LUX': 'Luxembourg', 'STHEL': 'Saint Helena', 'BHR': 'Bahrain', 'KSA': 'Saudi Arabia', 'MLT': 'Malta', 'Arg': 'Argentina', 'MNG': 'Mongolia', 'AUS': 'Australia', 'GIBR': 'Gibraltar', 'SGP': 'Singapore', 'Chile': 'Chile', 'UAE': 'United Arab Emirates', 'NZ': 'New Zealand', 'SCOT': 'Scotland', 'BHU': 'Bhutan', 'MAS': 'Malaysia', 'BOT': 'Botswana', 'CRC': 'Costa Rica', 'PHI': 'Philippines', 'NAM': 'Namibia', 'RWN': 'Rwanda', 'OMA': 'Oman', 'NOR': 'Norway', 'CRT': 'Croatia', 'SWE': 'Sweden', 'Iran': 'Iran', 'PORT': 'Portugal', 'NED': 'Netherlands', 'SA': 'South Africa', 'SVN': 'Slovenia', 'GUE': 'Guernsey', 'MDV': 'Maldives', 'BHM': 'Bahamas', 'SWE': 'Sweden', 'MLT': 'Malta', 'ITA': 'Italy',
        }

    try:
        bowling_data = load_data(spark, raw_data_dir, 't20_bowling_stats.csv')
        bowling_data = preprocess_bowling_data(bowling_data)
        bowling_data = map_country_codes(bowling_data, country_codes)
        save_data(bowling_data, processed_data_dir, 'bowling_data.csv')
        logging.info("Bowling data processing and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in preprocess_bowling task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

def preprocess_fielding():
    logging.info("Starting preprocess_fielding task.")
    spark = create_spark_session()
    base_dir = "hdfs://192.168.245.142:8020/usr/ravi/t20"
    raw_data_dir = os.path.join(base_dir, 'data', '1_rawData')
    processed_data_dir = os.path.join(base_dir, 'data', '2_processedData')

    country_codes = {
            'LES': 'Lesotho', 'BUL': 'Bulgaria', 'VAN': 'Vanuatu', 'ROM': 'Romania', 'Aut': 'Austria', 'COK': 'Cook Islands', 'Fran': 'France', 'SRB': 'Serbia', 'PAK': 'Pakistan', 'HUN': 'Hungary', 'CYP': 'Cyprus', 'Fiji': 'Fiji', 'FIN': 'Finland', 'EST': 'Estonia', 'CHN': 'China', 'GRC': 'Greece', 'CAM': 'Cambodia', 'GUE': 'Guernsey', 'SEY': 'Seychelles', 'JPN': 'Japan', 'TAN': 'Tanzania', 'JER': 'Jersey', 'QAT': 'Qatar', 'ENG': 'England', 'UGA': 'Uganda', 'BER': 'Bermuda', 'CZK-R': 'Czech Republic', 'CAY': 'Cayman Islands', 'IRE': 'Ireland', 'Mali': 'Mali', 'BRA': 'Brazil', 'SUI': 'Switzerland', 'Peru': 'Peru', 'Mex': 'Mexico', 'MOZ': 'Mozambique', 'Samoa': 'Samoa', 'HKG': 'Hong Kong', 'BAN': 'Bangladesh', 'SL': 'Sri Lanka', 'PNG': 'Papua New Guinea', 'ZIM': 'Zimbabwe', 'GHA': 'Ghana', 'SWZ': 'Eswatini', # Swaziland's official name now is Eswatini
            'MYAN': 'Myanmar', 'IND': 'India', 'USA': 'United States of America', 'NEP': 'Nepal', 'AFG': 'Afghanistan', 'PAN': 'Panama', 'NGA': 'Nigeria', 'SLE': 'Sierra Leone', 'ESP': 'Spain', 'Bhm': 'Bahamas', 'TKY': 'Turkey', 'MWI': 'Malawi', 'WI': 'West Indies', 'IOM': 'Isle of Man', 'THA': 'Thailand', 'SWA': 'Eswatini',
            'SKOR': 'South Korea', 'GMB': 'Gambia', 'ISR': 'Israel', 'KUW': 'Kuwait', 'Belg': 'Belgium', 'GER': 'Germany', 'ITA': 'Italy', 'CAN': 'Canada', 'MDV': 'Maldives', 'Blz': 'Belize', 'DEN': 'Denmark', 'INA': 'Indonesia', 'KENYA': 'Kenya', 'LUX': 'Luxembourg', 'STHEL': 'Saint Helena', 'BHR': 'Bahrain', 'KSA': 'Saudi Arabia', 'MLT': 'Malta', 'Arg': 'Argentina', 'MNG': 'Mongolia', 'AUS': 'Australia', 'GIBR': 'Gibraltar', 'SGP': 'Singapore', 'Chile': 'Chile', 'UAE': 'United Arab Emirates', 'NZ': 'New Zealand', 'SCOT': 'Scotland', 'BHU': 'Bhutan', 'MAS': 'Malaysia', 'BOT': 'Botswana', 'CRC': 'Costa Rica', 'PHI': 'Philippines', 'NAM': 'Namibia', 'RWN': 'Rwanda', 'OMA': 'Oman', 'NOR': 'Norway', 'CRT': 'Croatia', 'SWE': 'Sweden', 'Iran': 'Iran', 'PORT': 'Portugal', 'NED': 'Netherlands', 'SA': 'South Africa', 'SVN': 'Slovenia', 'GUE': 'Guernsey', 'MDV': 'Maldives', 'BHM': 'Bahamas', 'SWE': 'Sweden', 'MLT': 'Malta', 'ITA': 'Italy',
        }

    try:
        fielding_data = load_data(spark, raw_data_dir, 't20_fielding_stats.csv')
        fielding_data = preprocess_fielding_data(fielding_data)
        fielding_data = map_country_codes(fielding_data, country_codes)
        save_data(fielding_data, processed_data_dir, 'fielding_data.csv')
        logging.info("Fielding data processing and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in preprocess_fielding task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")

def combine_data():
    logging.info("Starting combine_data task.")
    spark = create_spark_session()
    processed_data_dir = "hdfs://192.168.245.142:8020/usr/ravi/t20/data/2_processedData"

    try:
        batting_data = load_data(spark, processed_data_dir, 'batting_data.csv')
        bowling_data = load_data(spark, processed_data_dir, 'bowling_data.csv')
        fielding_data = load_data(spark, processed_data_dir, 'fielding_data.csv')
        players_data = load_data(spark, processed_data_dir, 'Players.csv')

        batting_data = batting_data.join(players_data, ['Player', 'Country'], 'inner')
        bowling_data = bowling_data.join(players_data, ['Player', 'Country'], 'inner')
        fielding_data = fielding_data.join(players_data, ['Player', 'Country'], 'inner')

        bowling_data = bowling_data.drop('Mat', 'Inns')
        fielding_data = fielding_data.drop('Mat', 'Inns')

        player_data = batting_data.join(
            bowling_data, on=['player_id', 'Player', 'Country', 'Season'], how='inner'
        ).join(
            fielding_data, on=['player_id', 'Player', 'Country', 'Season'], how='inner'
        ).drop('Cumulative Mat', 'Cumulative Inns')

        save_data(player_data, processed_data_dir, 'playerstats.csv')
        logging.info("Data combining and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error in combine_data task: {e}")
        raise
    finally:
        spark.stop()
        logging.info("Spark session stopped.")