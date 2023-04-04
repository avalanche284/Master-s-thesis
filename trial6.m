%
%
%%
%%%
% This script has both versions: with normalizaiotn and without it.
% It is intended to be run concurrently so that error measurements and
% plots in the end show differences in prediction accuracy. However, if you
% prefer cleaner output meaning -- without unnecessary graphs -- just
% comment/uncomment what is needed. Part of the script that relate to
% normalization are marked as below.

% ### Normalization: This block of code applies normalization to the dataset
%%%
%%
%
%
%% karkat (C) AW 2021 
%% Szymon Bartoszewicz WSB Merito University in Gdansk, 2023
% The purpose of this script is to forecast PM2.5 and PM10 concurrently
% based on predictors: pm2_5	pm10	o3	no2	so2	co	temp	pressure
% humidity	wind_speed	clouds.
% Pollutant concentration in μg/m3. Units:temp -- kelvin, pressure -- hPa,
% humidity -- %, wind_speed -- meter/sec, clouds -- %.
% The dataset comes from OpenWeather® --> apd21_air_pollution.py
% & apd51_weatehr_data.py
% Datasets were combined to get daily averages of pollutants and weather
% conditions i.e., temp	pressure	humidity	wind_speed	clouds	pm2_5
% pm10	o3	no2	so2	co.
% The script uses the sliding window approach and Moore-Penrose
% pseudoinverse.Additionally, it contains error measures such as R-squared
% and MAPE RMSE, MSE, MAD.
%

clear all
% Load data from CSV file
filename = 'warsaw_data.csv';
T = readtable(filename); % Read data from CSV file into a table
C = T{:, 2:11}; % Extract the columns without "date"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Descriptive statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = size(C);
summary(T)
column_names = T.Properties.VariableNames(2:11); % Extract the column names
% from the table T

% Histograms
% It's a diagram consisting of rectangles whose area is proportional to the
% frequency of a variable and whose width is equal to the class interval.
for i = 1:size(C, 2)
    figure;
    histogram(C(:, i));
    xlabel(column_names{i});
    ylabel('Frequency');
    title(['(Before tranformation)Histogram of ' column_names{i}]);
 end


% ### Normalization
% After examination of above histogram charts, I state that:
% data of PM2.5, PM10, NO2, SO2, CO, wind_speed is left-skewed,
% data of humidity is right-skewed.
% Having the above in mind, cube and log transformations will be applied
% sequentially.

% Addition a small constant to handle zeros in the data (if any) for log transformation
constant = 1;
% Initialization the transformed dataset with the original data
C_transformed = C;
% Log transformation to right-skewed features
C_transformed(:, [1, 2, 4, 5, 6, 10]) = log(C(:, [1, 2, 4, 5, 6, 10]) + constant);
% Previous tests shew O_3's data durability to log tranformation. That is
% why square root transformation is utilized in this case.
C_transformed(:, [3]) = sqrt(C(:, [3]) + constant);
% Cube transformation to left-skewed features
C_transformed(:, [9]) = C(:, [9]).^3;
for i = 1:size(C_transformed, 2)
    figure;
    histogram(C_transformed(:, i));
    xlabel(column_names{i});
    ylabel('Frequency');
    title(['Histogram of transformed' column_names{i}]);
end
% ### Normalization: End

% Scatter plot matrix
% A scatter plot matrix is a grid (or matrix) of scatter plots used to
% visualize bivariate relationships between combinations of variables. Each
% scatter plot in the matrix visualizes the relationship between a pair
% of variables.
figure;
plotmatrix(C);
title('Scatter Plot Matrix of Variables');

% Box Plot
% A boxplot is a standardized way of displaying the distribution of data
% based on a five number summary (“minimum”, first quartile [Q1], median,
% third quartile [Q3] and “maximum”). It can tell you about your outliers
% and what their values are.
figure;
boxplot(C, 'Labels', column_names);
title('Box Plot of Variables');

% Standard deviation
% The standard deviation helps to understand how much the individual data
% points differ from the mean of the dataset.
std_dev = std(C);
fprintf('\nStandard Deviation for each feature:\n');
for i = 1:length(column_names)
    fprintf('%s: %.4f\n', column_names{i}, std_dev(i));
end

% Correlation plots
% Predictors should be chosen such that they have high correlaton with the
% response variable and low correlation between each other (to avoid
% redundancy).
column_names = T.Properties.VariableNames(2:11);
figure
correlation_plot = corrplot(C, 'varNames', column_names);
title('Correlation plot');

% Heatmap
% Correlation heatmaps are a type of plot that visualize the strength of
% relationships between numerical variables.
correlation_matrix = corrcoef(C);
figure;
heatmap(column_names, column_names, correlation_matrix);
title('Heatmap of Correlation Matrix');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


window=16; % window size -- a number of data points taken from the past to
% make edicitons
kkk=100;% the number of maximum windows that can be used 
hp=1; % a prediction horizon
beg1 = m(1) - kkk - window - hp; %represents the starting index for the first window in the sliding window approach

% ### Normalization:
m_transformed = size(C_transformed);
beg1 = m_transformed(1) - kkk - window - hp; %represents the starting index for the first window in the sliding window approach
% ### Normalization:

for i = 1:kkk-1
    beg = beg1 + i;
    fin = beg + window;

    Cn(i, :) = C(fin + hp + 1, 1:2);
    C1 = C(beg:fin-1, [3 4 5 6]); 
    y = C(beg + hp:fin + hp - 1, 1:2)';
    X = [ones(size(y, 2), 1), C1];
    A = X \ y'; % the Moore-Penrose pseudoinverse 
    Ym = X * A;
    dely = y - Ym';
    Xlast = [1, C(fin + hp + 1, [3 4 5 6])];
    ynext(i, :) = Xlast * A;
    error(i, :) = abs(ynext(i, :) - C(fin + hp + 1, 1:2));
    Ai(i, :, :) = A;

    % ### Normalization:
    Cn_transformed(i, :) = C_transformed(fin + hp + 1, 1:2);
    C1_transformed = C_transformed(beg:fin-1, [3 4 5 6]); 
    y_transformed = C_transformed(beg + hp:fin + hp - 1, 1:2)';
    X_transformed = [ones(size(y_transformed, 2), 1), C1_transformed];
    A_transformed = X_transformed \ y_transformed'; % the Moore-Penrose pseudoinverse 
    Ym_transformed = X_transformed * A_transformed;
    dely_transformed = y_transformed - Ym_transformed';
    Xlast_transformed = [1, C_transformed(fin + hp + 1, [3 4 5 6])];
    ynext_transformed(i, :) = Xlast_transformed * A_transformed;
    error_transformed(i, :) = abs(ynext_transformed(i, :) - C_transformed(fin + hp + 1, 1:2));
    Ai_transformed(i, :, :) = A_transformed;
end

% Error metrics calculation
MAE = mean(error, 1);
MSE = mean(error.^2, 1);
RMSE = sqrt(MSE);
MAPE = mean(abs(error ./ Cn), 1) * 100;
MAD = median(abs(error), 1);
% R-squared
SS_total = sum((Cn - mean(Cn)).^2);
SS_residual = sum(error.^2);
R_squared = 1 - (SS_residual ./ SS_total);

% Display error metrics and R-squared
fprintf('PM2.5 Prediction Metrics:\n');
fprintf('MAE: %.4f\n', MAE(1));
fprintf('MSE: %.4f\n', MSE(1));
fprintf('RMSE: %.4f\n', RMSE(1));
fprintf('R-squared: %.4f\n', R_squared(1));
fprintf('MAPE: %.4f%%\n', MAPE(1));
fprintf('MAD: %.4f\n', MAD(1));

fprintf('\nPM10 Prediction Metrics:\n');
fprintf('MAE: %.4f\n', MAE(2));
fprintf('MSE: %.4f\n', MSE(2));
fprintf('RMSE: %.4f\n', RMSE(2));
fprintf('R-squared: %.4f\n', R_squared(2));
fprintf('MAPE: %.4f%%\n', MAPE(2));
fprintf('MAD: %.4f\n', MAD(2));

figure
plot(30:99, ynext(30:99, 1), '-r')
hold on
plot(30:99, Cn(30:99, 1), '-g')
title('Model (red) and real (green) for PM2.5')

figure
plot(30:99, ynext(30:99, 2), '-r')
hold on
plot(30:99, Cn(30:99, 2), '-g')
title('Model (red) and real (green) for PM10')





% Initialize the inverse transformed predictions with the transformed predictions
ynext_inverse_transformed = ynext;

% Apply the inverse of the log transformation (exponential function) to the features that were log-transformed
ynext_inverse_transformed(:, [1, 2]) = exp(ynext(:, [1, 2])) - constant;

%ynext_inverse_transformed(:, [3]) = (ynext(:, [3]).^2) - constant;

% Apply the inverse of the cube transformation (cube root) to the features that were cube-transformed
%ynext_inverse_transformed(:, [9]) = ynext(:, [9]).^(1/3);


error_inverse_transformed = abs(ynext_inverse_transformed - Cn);

% Error metrics calculation
MAE_inverse_transformed = mean(error_inverse_transformed, 1);
MSE_inverse_transformed = mean(error_inverse_transformed.^2, 1);
RMSE_inverse_transformed = sqrt(MSE_inverse_transformed);
MAPE_inverse_transformed = mean(abs(error_inverse_transformed ./ Cn), 1) * 100;
MAD_inverse_transformed = median(abs(error_inverse_transformed), 1);

% Display error metrics for inverse transformed predictions
fprintf('\nInverse Transformed PM2.5 Prediction Metrics:\n');
fprintf('MAE: %.4f\n', MAE_inverse_transformed(1));
fprintf('MSE: %.4f\n', MSE_inverse_transformed(1));
fprintf('RMSE: %.4f\n', RMSE_inverse_transformed(1));
fprintf('MAPE: %.4f%%\n', MAPE_inverse_transformed(1));
fprintf('MAD: %.4f\n', MAD_inverse_transformed(1));

fprintf('\nInverse Transformed PM10 Prediction Metrics:\n');
fprintf('MAE: %.4f\n', MAE_inverse_transformed(2));
fprintf('MSE: %.4f\n', MSE_inverse_transformed(2));
fprintf('RMSE: %.4f\n', RMSE_inverse_transformed(2));
fprintf('MAPE: %.4f%%\n', MAPE_inverse_transformed(2));
fprintf('MAD: %.4f\n', MAD_inverse_transformed(2));

figure
plot(30:99, ynext_inverse_transformed(30:99, 1), '-r')
hold on
plot(30:99, Cn(30:99, 1), '-g')
title('Inverse Transformed Model (red) and Real (green) for PM2.5')

figure
plot(30:99, ynext_inverse_transformed(30:99, 2), '-r')
hold on
plot(30:99, Cn(30:99, 2), '-g')
title('Inverse Transformed Model (red) and Real (green) for PM10')







% Calculate the correlation coefficient between PM2.5 and PM10
figure
correlation_coefficient = corr(C(:, 1), C(:, 2));
fprintf('Correlation Coefficient between PM2.5 and PM10: %.4f\n', correlation_coefficient);

% Calculate the partial correlation coefficients (controlling for all other variables)
partial_correlation_matrix = partialcorr(C);

% Draw the partial correlation graph
partial_corr_plot(partial_correlation_matrix, column_names);

% Bar Plot for Error Metrices
% Not sure whther it gives a useful piece of information.
error_metrics = [MAE; MSE; RMSE; R_squared; MAPE; MAD];
error_metric_names = {'MAE', 'MSE', 'RMSE', 'R-squared', 'MAPE', 'MAD'};
figure;
bar(error_metrics);
set(gca, 'XTickLabel', error_metric_names);
legend('PM2.5', 'PM10');
title('Error Metrics for PM2.5 and PM10 Predictions');

