from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Cricket-Prediction").getOrCreate()

directory = r'D:\github\Cricket-Prediction\data\2_processedData'

# Load the data
teams = spark.read.csv(directory + r'\teamStats.csv', header=True, inferSchema=True)
matches = spark.read.csv(directory + r'\matches.csv', header=True, inferSchema=True)
teams.show(5)

tdt = teams.select("Team").distinct().rdd.map(lambda row: row.Team).collect()
mdt = matches.select("Team1").distinct().rdd.map(lambda row: row.Team1).collect()

for i in tdt:
    if i not in mdt:
        print(i)

for i in mdt:
    if i not in tdt:
        print(i)


# Teams that need to be mapped between `tdt` and `mdt`
team_name_mapping = {
    'U.S.A.': 'United States of America',
    'U.A.E.': 'United Arab Emirates',
    'Czech Rep.': 'Czech Republic',
    'P.N.G.': 'Papua New Guinea',
    'Cayman': 'Cayman Islands'
}

# Teams that do not have a direct match
unmatched_tdt = [team for team in tdt if team not in mdt and team not in team_name_mapping]
unmatched_mdt = [team for team in mdt if team not in tdt and team not in team_name_mapping.values()]

print("Mapped Team Names Dictionary:", team_name_mapping)
print("Unmatched Teams in tdt:", unmatched_tdt)
print("Unmatched Teams in mdt:", unmatched_mdt)

unmatched_teams = unmatched_tdt + unmatched_mdt
unmatched_teams


print(teams.count(), matches.count())
teams = teams.filter(~teams.Team.isin(unmatched_teams))
matches = matches.filter(~matches.team1.isin(unmatched_teams)).filter(~matches.team2.isin(unmatched_teams))
print(teams.count(), matches.count())


teams = teams.replace(team_name_mapping, subset='Team')
matches = matches.replace(team_name_mapping, subset='team1').replace(team_name_mapping, subset='team2')


from pyspark.sql import functions as F

matches1 = matches
matches1 = matches1.withColumn('flip', F.lit(0))
matches2 = matches.withColumnRenamed('team1', 'temp_team').withColumnRenamed('team2', 'team1').withColumnRenamed('temp_team', 'team2').select(
    ['team1', 'team2', 'gender', 'season', 'date', 'venue', 'city', 'toss_winner', 'toss_decision', 'winner','match_id'])
matches2 = matches2.withColumn('flip', F.lit(1))
matchesflip = matches1.union(matches2).sort('match_id')
matchesflip.show(5)


matchesflip.join(teams, on=[matchesflip.team1 == teams.Team, matchesflip.season == teams.Season], how='inner').drop("Team",teams.Season).show(5)

# matchesflip.join(team_data, left_on=['team1','season'], right_on=['Team','Season'], how='inner',suffix='_team1')

matchesflip = matchesflip.join(teams, on=[matchesflip.team1 == teams.Team, matchesflip.season == teams.Season], how='inner').drop("Team",teams.Season)
matchesflip = matchesflip.withColumnsRenamed({
    "Cumulative Won": "Cumulative Won team1",
    "Cumulative Lost": "Cumulative Lost team1",
    "Cumulative Tied": "Cumulative Tied team1",
    "Cumulative NR": "Cumulative NR team1",
    "Cumulative W/L": "Cumulative W/L team1",
    "Cumulative AveRPW": "Cumulative AveRPW team1", 
    "Cumulative AveRPO": "Cumulative AveRPO team1", 
})
matchesflip.show(5)


matchesflip.join(teams, on=[matchesflip.team2 == teams.Team, matchesflip.season == teams.Season], how='inner').drop("Team",teams.Season).show(5)


teams_renamed = teams.withColumnRenamed("Season", "Team_Season")

matchesflip = matchesflip.join(teams_renamed, on=[matchesflip.team2 == teams_renamed.Team, matchesflip.season == teams_renamed.Team_Season], how='inner').drop("Team", "Team_Season")
matchesflip = matchesflip.withColumnsRenamed({
    "Cumulative Won": "Cumulative Won team2",
    "Cumulative Lost": "Cumulative Lost team2",
    "Cumulative Tied": "Cumulative Tied team2",
    "Cumulative NR": "Cumulative NR team2",
    "Cumulative W/L": "Cumulative W/L team2",
    "Cumulative AveRPW": "Cumulative AveRPW team2",
    "Cumulative AveRPO": "Cumulative AveRPO team2",
})
matchesflip.show(5)

# male 0 female 1

matchesflip = matchesflip.withColumn("gender", F.when(matchesflip['gender']=="male",0).otherwise(1).cast("int"))
matchesflip.show(5)


# match_id|flip|gender| season|      date|               venue|       city|toss_winner|toss_decision|   winner|Cumulative Won team1|Cumulative Lost team1|Cumulative Tied team1|Cumulative NR team1|Cumulative W/L team1|Cumulative AveRPW team1|Cumulative AveRPO team1|Cumulative Won team2|Cumulative Lost team2|Cumulative Tied team2|Cumulative NR team2|Cumulative W/L team2|Cumulative AveRPW team2|Cumulative AveRPO team2|

matchesflip = matchesflip.select("match_id","flip","gender","Cumulative Won team1","Cumulative Lost team1","Cumulative Tied team1","Cumulative NR team1","Cumulative W/L team1","Cumulative AveRPW team1","Cumulative AveRPO team1","Cumulative Won team2","Cumulative Lost team2","Cumulative Tied team2","Cumulative NR team2","Cumulative W/L team2","Cumulative AveRPW team2","Cumulative AveRPO team2").sort("match_id",'flip')
matchesflip.show(5)

directory = r'D:\github\Cricket-Prediction\data\3_aftermerging'

matchesflip.toPandas().to_csv(directory + r'\team12Statsflip.csv', index=False)
