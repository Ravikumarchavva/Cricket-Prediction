import os
import glob
import polars as pl
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, sum, when

def preprocess_team_stats():
    return None

def process_players_data():
    # ...existing code from playersdata.py...
    # Specify the directory where your CSV files are located
    directory = r'D:\github\Cricket-Prediction\data\1_rawData\t20s_csv2'

    # Use glob to find all CSV files in the specified directory
    info_files = glob.glob(os.path.join(directory, '*_info.csv'))
    all_files = glob.glob(os.path.join(directory,'*.csv'))
    delivery_files = [file for file in all_files if '_info' not in file]

    matches=[]
    deliveries=[]
    # Print the list of CSV files
    for info_file in info_files:
        matches.append(info_file.split('\\')[-1])
    for delivery in delivery_files:
        if '_info' not in delivery:
            deliveries.append(delivery.split('\\')[-1])

    dataframes = pd.DataFrame(columns=['country', 'player','player_id','season','match_id'])
    injured_matches = []

    for info_file in info_files:
        match_id = pd.to_numeric(info_file.split('\\')[-1].split('_')[0])
        try:
            df = pd.read_csv(info_file, header=None, names=['type', 'heading', 'subkey', 'players','player_id'], skipinitialspace=True).drop('type', axis=1)
            players_df = df[df['heading'] == "player"].drop(['heading','player_id'], axis=1)
            registry_df = df[df['heading'] == "registry"].drop('heading', axis=1)
            merged_df = players_df.merge(registry_df[['players', 'player_id']], on='players', how='inner')
            merged_df.rename(columns={'players':'player','subkey':'country'}, inplace=True)
            season = df['subkey'][5] 
            merged_df['match_id'] = match_id
            merged_df['season'] = season
            if(len(merged_df)!=22):
                raise Exception('Injured Match')
            dataframes = pd.concat([dataframes, merged_df])
        except:
            injured_matches.append(match_id)
    print(injured_matches)

    dataframes.to_csv(os.path.join(directory,"../../2_processedData/Matchplayers.csv"),index=False)

    players = pl.from_pandas(dataframes).drop('match_id').select('player','country','player_id').unique()
    players.write_csv(os.path.join(directory,'../../2_processedData/Players.csv'))

def process_matches_data():
    # ...existing code from matchesdata.py...
    # Specify the directory where your CSV files are located
    directory = r'D:\github\Cricket-Prediction\data\1_rawData\t20s_csv2'

    # Use glob to find all CSV files in the specified directory
    info_files = glob.glob(os.path.join(directory, '*_info.csv'))
    all_files = glob.glob(os.path.join(directory,'*.csv'))
    delivery_files = [file for file in all_files if '_info' not in file]

    matches=[]
    deliveries=[]
    # Print the list of CSV files
    for info_file in info_files:
        matches.append(info_file.split('\\')[-1])
    for delivery in delivery_files:
        if '_info' not in delivery:
            deliveries.append(delivery.split('\\')[-1])

    match_ids=[]
    for csv_file in matches:
        match_ids.append(csv_file.split('_')[0])
        
    # Define the initial and final schemas
    initial_schema = {'col1': pl.String, 'attributes': pl.String, 'values': pl.String, 'players': pl.String, 'code': pl.String}
    final_schema = [
        ('team1', pl.String),
        ('team2', pl.String),
        ('gender', pl.String),
        ('season', pl.String),
        ('date', pl.String),
        ('venue', pl.String),
        ('city', pl.String),
        ('toss_winner', pl.String),
        ('toss_decision', pl.String),
        ('winner', pl.String),
    ]

    # Create a dictionary from the final schema
    final_schema_dict = {key: value for key, value in final_schema}

    # Initialize an empty DataFrame with the final schema
    matches_data = pl.DataFrame(schema=final_schema_dict)

    # List to store recalculated match IDs
    recalculated_matchids = match_ids[:]
    import tqdm
    # Iterate over matches and process each one
    for idx, match in enumerate(tqdm.tqdm(matches)):
        try:
            match_df = pl.read_csv(os.path.join(directory,f'{match}'), schema=initial_schema)
            # Extract team names
            team1_name = match_df[1, 'values']
            team2_name = match_df[2, 'values']
            
            # Replace team names
            match_df = match_df.with_columns([
                pl.when((pl.col('attributes') == 'team') & (pl.col('values') == team1_name))
                .then(pl.lit('team1'))
                .when((pl.col('attributes') == 'team') & (pl.col('values') == team2_name))
                .then(pl.lit('team2'))
                .otherwise(pl.col('attributes'))
                .alias('attributes')
            ])
            
            # Select and transpose the DataFrame
            match_transposed = match_df.select("attributes", "values").transpose(include_header=True, column_names="attributes").drop("column")
            
            # Ensure all columns in final_schema_dict are present
            missing_cols = [col for col in final_schema_dict.keys() if col not in match_transposed.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")

            # Select the required columns and append to matches_data
            match_transposed = match_transposed.select(final_schema_dict.keys())
            matches_data = matches_data.vstack(match_transposed)
        except Exception as e:
            recalculated_matchids.remove(match_ids[idx])
    matches_data=matches_data.with_columns(pl.Series(recalculated_matchids).alias("match_id").cast(pl.Int64))
    matches_data.write_csv(os.path.join(directory, '../../2_processedData/matches.csv'))

def process_deliveries_data():
    # ...existing code from deliveriesdata.py...
    # Specify the directory where your CSV files are located
    directory = r'D:\github\Cricket-Prediction\data\1_rawData\t20s_csv2'

    # Use glob to find all CSV files in the specified directory
    info_files = glob.glob(os.path.join(directory, '*_info.csv'))
    all_files = glob.glob(os.path.join(directory,'*.csv'))
    delivery_files = [file for file in all_files if '_info' not in file]

    matches=[]
    deliveries=[]
    # Print the list of CSV files
    for info_file in info_files:
        matches.append(info_file.split('\\')[-1])
    for delivery in delivery_files:
        if '_info' not in delivery:
            deliveries.append(delivery.split('\\')[-1])

    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder.appName('deliveries').getOrCreate()

    # Define the schema for the deliveries data
    delivery_schema = StructType([
        StructField('match_id', IntegerType(), True),
        StructField('season', StringType(), True),
        StructField('start_date', StringType(), True),
        StructField('venue', StringType(), True),
        StructField('innings', IntegerType(), True),
        StructField('ball', FloatType(), True),
        StructField('batting_team', StringType(), True),
        StructField('bowling_team', StringType(), True),
        StructField('striker', StringType(), True),
        StructField('non_striker', StringType(), True),
        StructField('bowler', StringType(), True),
        StructField('runs_off_bat', IntegerType(), True),
        StructField('extras', IntegerType(), True),
        StructField('wides', IntegerType(), True),
        StructField('noballs', StringType(), True),
        StructField('byes', IntegerType(), True),
        StructField('legbyes', IntegerType(), True),
        StructField('penalty', StringType(), True),
        StructField('wicket_type', StringType(), True),
        StructField('player_dismissed', StringType(), True),
        StructField('other_wicket_type', StringType(), True),
        StructField('other_player_dismissed', StringType(), True)
    ])

    # Initialize an empty DataFrame with the schema
    deliveries_data = spark.read.csv(delivery_files, header=True, schema=delivery_schema)
    deliveries_data = deliveries_data.fillna(0)
    deliveries_data = deliveries_data.withColumn('noballs', when(col('noballs').isNull(), '0').otherwise(col('noballs')).cast(IntegerType()))
    deliveries_data = deliveries_data.withColumn('penalty', when(col('penalty').isNull(), '0').otherwise(col('penalty')).cast(IntegerType()))
    columns = ['wicket_type','player_dismissed','other_wicket_type','other_player_dismissed']
    for column in columns:
        deliveries_data = deliveries_data.withColumn(column, when(col(column).isNull(), '0').otherwise('1').cast(IntegerType()))

    deliveries_data.toPandas().to_parquet(os.path.join(r'D:\github\Cricket-Prediction\data\2_processedData','deliveries.parquet'),index=False)