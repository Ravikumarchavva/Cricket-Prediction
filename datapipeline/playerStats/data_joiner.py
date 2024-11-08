
import logging

def join_data(batting_data, bowling_data, fielding_data):
    """Join batting, bowling, and fielding data on specified keys."""
    logging.info("Joining batting, bowling, and fielding data.")
    # Drop unnecessary columns to avoid duplicates
    bowling_data = bowling_data.drop('Cumulative Mat', 'Cumulative Inns')
    fielding_data = fielding_data.drop('Cumulative Mat', 'Cumulative Inns')

    # Join DataFrames on specific columns
    player_data = batting_data.join(
        bowling_data, on=['player_id', 'Player', 'Country', 'Season'], how='inner'
    ).join(
        fielding_data, on=['player_id', 'Player', 'Country', 'Season'], how='inner'
    )

    return player_data
