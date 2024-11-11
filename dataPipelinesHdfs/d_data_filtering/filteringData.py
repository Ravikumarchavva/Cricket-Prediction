import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
import utils

def filter_data():

    spark = utils.create_spark_session("DataFiltering")

    team12Stats = utils.load_data(spark, config.MERGED_DATA_DIR, 'team12Statsflip.csv')
    balltoball = utils.load_data(spark, config.MERGED_DATA_DIR, 'balltoballflip.csv')
    playersStats = utils.load_data(spark, config.MERGED_DATA_DIR, 'playersStatsflip.csv')

    # Extract match_id columns from each dataset
    team12_match_ids = team12Stats.select('match_id').distinct()
    balltoball_match_ids = balltoball.select('match_id').distinct()
    player_match_ids = playersStats.select('match_id').distinct()

    # Find intersection of match_id across all three datasets
    common_match_ids = team12_match_ids.intersect(balltoball_match_ids).intersect(player_match_ids)

    # convert to list
    common_match_ids_list = [row['match_id'] for row in common_match_ids.collect()]

    # filter datasets by common match_ids
    filtered_team12Stats = team12Stats.filter(team12Stats.match_id.isin(common_match_ids_list)).drop("_c0")
    filtered_balltoball = balltoball.filter(balltoball.match_id.isin(common_match_ids_list)).drop("_c0")
    filtered_playersStats = playersStats.filter(playersStats.match_id.isin(common_match_ids_list)).drop("_c0")

    filtered_balltoball = filtered_balltoball.select('match_id','flip','innings','ball','curr_score','curr_wickets','target',"won")

    # Save filtered datasets to HDFS
    utils.save_data(filtered_team12Stats, config.FILTERED_DATA_DIR, 'team12Stats.csv')
    utils.save_data(filtered_balltoball, config.FILTERED_DATA_DIR, 'balltoball.csv')
    utils.save_data(filtered_playersStats, config.FILTERED_DATA_DIR, 'playersStats.csv')

