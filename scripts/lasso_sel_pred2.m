% lasso2
import('matlab.io.*');

filename = 'warsaw_data.csv';
dataTable = readtable(filename);

dataTable.date = []; % Remove the date column
dataTable.pm10 = []; % Remove the pm10 column

splitRatio = 0.7; % 70% for training, 30% for testing
splitIndex = floor(height(dataTable) * splitRatio);
trainData = dataTable(1:splitIndex, :);
testData = dataTable(splitIndex + 1:end, :);

predictors = trainData{:, 2:end}; % All columns except pm2_5
response = trainData.pm2_5;

% Perform LASSO regression with cross-validation
[B, FitInfo] = lasso(predictors, response, 'CV', 10); % 10-fold cross-validation

% Analyze the results
lassoPlot(B, FitInfo, 'PlotType', 'Lambda', 'XScale', 'log');

% Find the optimal lambda and the corresponding coefficients
optimalLambdaIndex = FitInfo.Index1SE; % One standard error rule
optimalLambda = FitInfo.Lambda1SE;
optimalB = B(:, optimalLambdaIndex);
selectedPredictors = find(optimalB ~= 0); % Indices of the selected predictors

% Display the information
disp('Optimal Lambda:');
disp(optimalLambda);

disp('LASSO Coefficients for the optimal Lambda:');
disp(array2table(optimalB, 'RowNames', trainData.Properties.VariableNames(1, 2:end)));

disp('Selected predictors:');
disp(trainData.Properties.VariableNames(1, selectedPredictors + 1));

% Calculate adjusted R-squared for the selected predictors
testPredictors = testData{:, selectedPredictors + 1};
testResponse = testData.pm2_5;
predictedResponse = [ones(size(testPredictors, 1), 1), testPredictors] * [FitInfo.Intercept(optimalLambdaIndex); optimalB(selectedPredictors)];
SST = sum((testResponse - mean(testResponse)).^2);
SSR = sum((testResponse - predictedResponse).^2);
adjustedR2 = 1 - (SSR/SST) * ((length(testResponse) - 1) / (length(testResponse) - length(selectedPredictors) - 1));

disp('Adjusted R-squared:');
disp(adjustedR2);
