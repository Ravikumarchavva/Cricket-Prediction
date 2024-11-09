#!/usr/bin/env python
# coding: utf-8

# # Adding paths and files

# In[1]:


import os
import glob
import polars as pl

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


# In[2]:


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
matches_data


# In[3]:


matches_data.null_count()


# In[4]:


matches_data.write_csv(os.path.join(directory, '../../2_processedData/matches.csv'))


# In[ ]:



