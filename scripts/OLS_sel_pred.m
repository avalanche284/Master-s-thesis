% OLS_sel_pred.m
% The sole purpose of this script is to show the most appropriate
% predictors to use. The cript uses OLS model
%% Szymon Bartoszewicz WSB Merito University in Gdansk, 2023

filename = 'warsaw_data.csv';
data = readtable(filename);
data.date = datetime(data.date);
data.pm10 = [];

predictors = data(:, 3:end); % Exclude date and PM2.5 columns
response = data(:, 2); % PM2.5 column
predictors_and_response = [predictors, response];

% LINEAR REGRESSION
formula = 'pm2_5 ~ o3 + no2 + so2 + co + temp + pressure + humidity + wind_speed + clouds';
mdl = fitlm(predictors_and_response, formula);

% Evaluation of the model
disp('Model Summary:');
disp(mdl);
disp('Model Coefficients:');
disp(mdl.Coefficients);
disp('Model R-squared:');
disp(mdl.Rsquared);
disp('Model Adjusted R-squared:');
disp(mdl.Rsquared.Adjusted);
disp('Model ANOVA:');
disp(anova(mdl));

% disp('P-values for each predictor:');
% disp(mdl.Coefficients.pValue);
