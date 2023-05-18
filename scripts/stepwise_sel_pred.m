% stepwise_sel_pred.m
% The sole purpose of this script is to select best predictors by showing
% the best model by use of a stepwise regression.
%% Szymon Bartoszewicz WSB Merito University in Gdansk, 2023

filename = 'warsaw_data.csv';
data = readtable(filename);

predictors = data(:, {'o3', 'no2', 'so2', 'co', 'temp', 'pressure', 'humidity', 'wind_speed', 'clouds'});
response = data.pm2_5;

tbl = [predictors, table(response)];

% Stepwise regression
mdl = stepwiselm(tbl, 'ResponseVar', 'response', 'PredictorVars', {'o3', 'no2', 'so2', 'co', 'temp', 'pressure', 'humidity', 'wind_speed', 'clouds'});

% Results
fprintf('Adjusted R-squared: %f\n', mdl.Rsquared.Adjusted);
fprintf('Best predictors:\n');
best_predictors = mdl.PredictorNames;
disp(best_predictors);
