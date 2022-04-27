# NBA_Game_Predictions_Using_ML
Website to test it out: 
https://nba-game-predicions.azurewebsites.net/

Allows the user to simulate an NBA season between 2014 and 2021.

Uses the first 70% of the season as training data and then uses the remaining 30% as test data.

Has a correct prediction rate of about 58%.

First 2 datasets are created with the season you selected.
The first dataset being all the games and their results.
And the second is all the teams and their current ELO rating.

Each team starts with a default ELO of 2000, when a team loses they lose 25 elo and when a team wins the gain 25.
The dataset is looped through and each game the ELO updates accordingly.

The ELO is used to help the model make the prediction.

All the data is collected using the NBA_api.

## Things to change 
1. Rework entire elo system ✓
2. Add end of season single game simulation	_
