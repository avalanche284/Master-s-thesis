import('matlab.io.*');

filename = 'warsaw_data.csv';
dataTable = readtable(filename);

dataTable.date = []; %Remove the date column
dataTable.pm10 = []; % Remove the pm10 column

splitRatio = 0.7; % 70% fortraining, 30% for testing
splitIndex = floor(height(dataTable) * splitRatio);
trainData = dataTable(1:splitIndex, :);
testData = dataTable(splitIndex + 1:end, :);

predictors = trainData{:, 2:end}; % All columns except pm2_5
response = trainData.pm2_5;

[B, FitInfo] = lasso(predictors, response, 'CV', 10); % 10-fold cross-validation

lassoPlot(B, FitInfo, 'PlotType', 'Lambda', 'XScale', 'log');

optimalLambdaIndex = FitInfo.Index1SE; % One standard errorule
optimalB = B(:, optimalLambdaIndex);
selectedPredictors = find(optimalB ~= 0); % Indices of the selected predictors

disp('Selected predictors:');
disp(trainData.Properties.VariableNames(1, selectedPredictors + 1));
