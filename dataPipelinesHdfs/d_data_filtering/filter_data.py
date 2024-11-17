import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
import utils

def filter_data():

    spark = utils.create_spark_session("DataFiltering")

    # Update data file names
    team12Stats = utils.load_data(spark, config.MERGED_DATA_DIR, 'team_stats.csv')
    balltoball = utils.load_data(spark, config.MERGED_DATA_DIR, 'ball_by_ball.csv')
    playersStats = utils.load_data(spark, config.MERGED_DATA_DIR, 'player_stats.csv')

    print(team12Stats.select('match_id').distinct().count(), balltoball.select('match_id').distinct().count(), playersStats.select('match_id').distinct().count())

    # Extract match_id columns from each dataset
    print("Extracting match_id columns from each dataset")
    team12_match_ids = team12Stats.select('match_id').distinct()
    balltoball_match_ids = balltoball.select('match_id').distinct()
    player_match_ids = playersStats.select('match_id').distinct()

    # Find intersection of match_id across all three datasets
    print("Finding intersection of match_id across all three datasets")
    common_match_ids = team12_match_ids.intersect(balltoball_match_ids).intersect(player_match_ids)

    # convert to list
    common_match_ids_list = [row['match_id'] for row in common_match_ids.collect()]

    # filter datasets by common match_ids
    print("Filtering datasets by common match_ids")
    filtered_team12Stats = team12Stats.filter(team12Stats.match_id.isin(common_match_ids_list)).drop("_c0")
    filtered_balltoball = balltoball.filter(balltoball.match_id.isin(common_match_ids_list)).drop("_c0")
    filtered_playersStats = playersStats.filter(playersStats.match_id.isin(common_match_ids_list)).drop("_c0")

    # Adjust columns selected in filtered_balltoball
    filtered_balltoball = filtered_balltoball.select(
        'match_id', 'innings', 'ball', 'runs', 'wickets',
        'curr_score', 'curr_wickets', 'overs', 'run_rate', 'required_run_rate',
        'target', 'won'
    )

    print(filtered_team12Stats.count(),"team12Stats,", filtered_balltoball.count(), "balltoball,", filtered_playersStats.count(), "playersStats")
    # Save filtered datasets to HDFS
    print("Saving filtered datasets to HDFS")
    utils.spark_save_data(filtered_team12Stats, config.FILTERED_DATA_DIR, 'team12_stats.csv')
    utils.spark_save_data(filtered_balltoball, config.FILTERED_DATA_DIR, 'ball_to_ball.csv')
    utils.spark_save_data(filtered_playersStats, config.FILTERED_DATA_DIR, 'players_stats.csv')

    # Stop the Spark session
    spark.stop()

if __name__ == '__main__':
    filter_data()