import matplotlib.pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import leaguestandingsv3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print(' 2014-15 = 1', '\n', '2015-16 = 2', '\n', '2016-17 = 3', '\n', '2017-18 = 4', '\n', '2018-19 = 5', '\n', '2019-20 = 6', '\n', '2020-21 = 7', '\n')
while True:
    try:
        season = int(input('Please select a season: '))
    except ValueError:
        print('Please enter a number')
        continue
    break

# DEFAULT SEASON
previous_season = '2013-14'
picked_season = '2014-15'

match season:
    case 1:
        previous_season = '2013-14'
        picked_season = '2014-15'
    case 2:
        previous_season = '2014-15'
        picked_season = '2015-16'
    case 3:
        previous_season = '2015-16'
        picked_season = '2016-17'
    case 4:
        previous_season = '2016-17'
        picked_season = '2017-18'
    case 5:
        previous_season = '2017-18'
        picked_season = '2018-19'
    case 6:
        previous_season = '2018-19'
        picked_season = '2019-20'
    case 7:
        previous_season = '2019-20'
        picked_season = '2020-21'
    case _:
        print('No season associated with', str(season) + ', defaulting to 2014-15 season')
        previous_season = '2013-14'
        picked_season = '2014-15'

print('Creating dataset from season you picked!')

all_games = leaguegamefinder.LeagueGameFinder(league_id_nullable='00', season_nullable=picked_season, season_type_nullable='Regular Season').get_data_frames()[0]
df_game_data = pd.DataFrame(all_games)
df_game_data = df_game_data.dropna()
df_game_data = df_game_data.drop_duplicates(subset=['GAME_ID'], keep='last')
df_game_data = df_game_data.drop(columns=['SEASON_ID', 'TEAM_ABBREVIATION', 'GAME_DATE', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA'
    , 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'TEAM_NAME', 'TEAM_ID'])

df_game_data['LEFT_ELO'] = 0
df_game_data['RIGHT_ELO'] = 0
df_game_data = df_game_data.iloc[::-1]

team_dict = teams.get_teams()
last_season_team_WL = leaguestandingsv3.LeagueStandingsV3(season=previous_season).get_data_frames()[0]
last_season_team_WL = pd.DataFrame(last_season_team_WL)
print('last season WL')
print(last_season_team_WL[['TeamName', 'WinPCT', 'TeamSlug']])

df_all_teams = pd.DataFrame(team_dict)
df_all_teams = df_all_teams.drop(columns=['id', 'nickname', 'city', 'state', 'year_founded'])
df_all_teams['ELO'] = 0
df_all_teams['Wins'] = 1
df_all_teams['Losses'] = 1
df_all_teams['Win Streak'] = 1
df_all_teams['Lose Streak'] = 1

print(df_all_teams)
print('Simulating season and adjusting elo...')

for index, row in last_season_team_WL.iterrows():
    for i, team in df_all_teams.iterrows():
        if row.loc['TeamName'] in df_all_teams.loc[i, 'full_name']:
            df_all_teams.loc[i, 'ELO'] = (df_all_teams.loc[i, 'ELO'] + (row.loc['WINS'] * 5)) - (row.loc['LOSSES'] * 5)

for index, row in df_game_data.iterrows():

    home_or_away = df_game_data.loc[index, 'MATCHUP']

    if df_game_data.loc[index, 'WL'] == 'L':
        losing_team = home_or_away[0:3]
        winning_team = home_or_away[-3:]
        recent_performance = 0

        for i in range(0, len(team_dict)):
            if df_all_teams.loc[i, 'abbreviation'] == losing_team:
                if df_game_data.loc[index, 'LEFT_ELO'] == 0:

                    if df_all_teams.loc[i, 'Lose Streak'] > 0:
                        recent_performance = 75 * df_all_teams.loc[i, 'Lose Streak']

                    df_game_data.loc[index, 'LEFT_ELO'] = df_all_teams.loc[i, 'ELO'] - recent_performance
                    df_all_teams.loc[i, 'ELO'] = df_all_teams.loc[i, 'ELO'] - 50 - round((35 * ((df_all_teams.loc[i, 'Wins']) / (df_all_teams.loc[i, 'Wins']
                                                                                           + df_all_teams.loc[i, 'Losses']))), 2)
                    df_all_teams.loc[i, 'Losses'] = df_all_teams.loc[i, 'Losses'] + 1
                    df_all_teams.loc[i, 'Lose Streak'] += 1
                    df_all_teams.loc[i, 'Win Streak'] = 0

            elif df_all_teams.loc[i, 'abbreviation'] == winning_team:
                if df_game_data.loc[index, 'RIGHT_ELO'] == 0:

                    if df_all_teams.loc[i, 'Win Streak'] > 0:
                        recent_performance = 75 * df_all_teams.loc[i, 'Win Streak']

                    df_game_data.loc[index, 'RIGHT_ELO'] = df_all_teams.loc[i, 'ELO'] + recent_performance
                    df_all_teams.loc[i, 'ELO'] = df_all_teams.loc[i, 'ELO'] + 50 + round((35 * ((df_all_teams.loc[i, 'Losses']) / (df_all_teams.loc[i, 'Wins']
                                                                                           + df_all_teams.loc[i, 'Losses']))), 2)
                    df_all_teams.loc[i, 'Wins'] = df_all_teams.loc[i, 'Wins'] + 1
                    df_all_teams.loc[i, 'Win Streak'] += 1
                    df_all_teams.loc[i, 'Lose Streak'] = 0

    else:
        winning_team = home_or_away[0:3]
        losing_team = home_or_away[-3:]
        recent_performance = 0

        for i in range(0, len(team_dict)):
            if df_all_teams.loc[i, 'abbreviation'] == winning_team:
                if df_game_data.loc[index, 'LEFT_ELO'] == 0:

                    if df_all_teams.loc[i, 'Win Streak'] > 0:
                        recent_performance = 75 * df_all_teams.loc[i, 'Win Streak']

                    df_game_data.loc[index, 'LEFT_ELO'] = df_all_teams.loc[i, 'ELO'] + recent_performance
                    df_all_teams.loc[i, 'ELO'] = df_all_teams.loc[i, 'ELO'] + 50 + round((35 * ((df_all_teams.loc[i, 'Losses']) / (df_all_teams.loc[i, 'Wins']
                                                                                           + df_all_teams.loc[i, 'Losses']))), 2)
                    df_all_teams.loc[i, 'Wins'] = df_all_teams.loc[i, 'Wins'] + 1
                    df_all_teams.loc[i, 'Win Streak'] += 1
                    df_all_teams.loc[i, 'Lose Streak'] = 0

            elif df_all_teams.loc[i, 'abbreviation'] == losing_team:
                if df_game_data.loc[index, 'RIGHT_ELO'] == 0:

                    if df_all_teams.loc[i, 'Lose Streak'] > 0:
                        recent_performance = 75 * df_all_teams.loc[i, 'Lose Streak']

                    df_game_data.loc[index, 'RIGHT_ELO'] = df_all_teams.loc[i, 'ELO'] - recent_performance
                    df_all_teams.loc[i, 'ELO'] = df_all_teams.loc[i, 'ELO'] - 50 - round((35 * ((df_all_teams.loc[i, 'Wins']) / (df_all_teams.loc[i, 'Wins']
                                                                                           + df_all_teams.loc[i, 'Losses']))), 2)
                    df_all_teams.loc[i, 'Losses'] = df_all_teams.loc[i, 'Losses'] + 1
                    df_all_teams.loc[i, 'Lose Streak'] += 1
                    df_all_teams.loc[i, 'Win Streak'] = 0

print('Training the model with the first 70% of the season...')
print(df_game_data)
X = df_game_data.drop(columns=['GAME_ID', 'MATCHUP', 'WL'])
y = df_game_data['WL']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, test_size=.3, shuffle=False)

print('Testing the last 30% of the season...')
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)

print('The model has predicted:', "{:.2%}".format(score) + '%', 'of the final 30% of games correct!', '\n')

df_game_data.to_csv('traindata.csv', sep='\t', encoding='utf-8')
df_all_teams.to_csv('teamelo.csv', sep='\t', encoding='utf-8')

# Displays graph of win loss percentages

wl = ['Wins', 'Losses']
slices = predictions
wrong = 1.00 - score
correct_incorrect = [score, wrong]
plt.pie(correct_incorrect,
        labels=wl,
        explode=(0.025, 0),
        autopct='%1.1f%%')

plt.title('Win Loss Graph')
plt.show()
