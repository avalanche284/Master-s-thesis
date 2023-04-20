% Step 1: Import necessary libraries
import('matlab.io.*');

% Step 2: Load the data from the CSV file
filename = 'warsaw_data.csv';
dataTable = readtable(filename);

% Step 3: Preprocess the data (optional)
dataTable.date = []; % Remove the date column
dataTable.pm10 = []; % Remove the pm10 column

% Step 4: Split the data into training and testing sets
splitRatio = 0.7; % 70% for training, 30% for testing
splitIndex = floor(height(dataTable) * splitRatio);
trainData = dataTable(1:splitIndex, :);
testData = dataTable(splitIndex + 1:end, :);

% Extract predictors and response
predictors = trainData{:, 2:end}; % All columns except pm2_5
response = trainData.pm2_5;

% Step 5: Perform LASSO regression with cross-validation
[B, FitInfo] = lasso(predictors, response, 'CV', 10); % 10-fold cross-validation

% Step 6: Analyze the results
lassoPlot(B, FitInfo, 'PlotType', 'Lambda', 'XScale', 'log');

% Find the optimal lambda and the corresponding coefficients
optimalLambdaIndex = FitInfo.Index1SE; % One standard error rule
optimalB = B(:, optimalLambdaIndex);
selectedPredictors = find(optimalB ~= 0); % Indices of the selected predictors

disp('Selected predictors:');
disp(trainData.Properties.VariableNames(1, selectedPredictors + 1));
