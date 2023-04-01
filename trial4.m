%% karkat (C) AW 2021 
% Szymon Bartoszewicz WSB Gdansk 2023
% The purpose of this script is to forecast PM2.5 and PM10 concurrently
% based on predictors such as pm2_5, pm10, o3, no2, so2, and co. 
% The dataset comes from OpenWeather® --> apd2.py & apd5.py
% Datasets were combined to get daily averages of pollutants and weather
% conditions i.e., temp	pressure	humidity	wind_speed	clouds	pm2_5
% pm10	o3	no2	so2	co.

% The script uses the sliding window approach and Moore-Penrose

% This script additionally contains erro measures such as R-squared and
% MAPE RMSE, MSE, MAD.
% Added corrplot
% Added partial correlation plot
% historgram
clear all
% Load data from CSV file
filename = 'warsaw_data.csv';
T = readtable(filename); % Read data from CSV file into a table
C = T{:, 2:11}; % Extract the columns with pm2_5, pm10, o3, no2, so2, co, temp, pressure, humidity, wind_speed, and clouds
m = size(C);

% Exploring the dataset
summary(T)
column_names = T.Properties.VariableNames(2:11); % Extract the column names from the table T

% Plot histograms for each variable
for i = 1:size(C, 2)
    figure;
    histogram(C(:, i));
    xlabel(column_names{i});
    ylabel('Frequency');
    title(['Histogram of ' column_names{i}]);
end

% Correlation plots
% Extract the column names from the table T
column_names = T.Properties.VariableNames(2:11);
% Display the correlation plot
figure
correlation_plot = corrplot(C, 'varNames', column_names);
title('Correlation plot');
% Heatmap
% Calculate the correlation matrix
correlation_matrix = corrcoef(C);
% Create a heatmap of the correlation matrix
figure;
heatmap(column_names, column_names, correlation_matrix);
title('Heatmap of Correlation Matrix');

% Scatter plot matrix
figure;
plotmatrix(C);
title('Scatter Plot Matrix of Variables');

% Box Plot
figure;
boxplot(C, 'Labels', column_names);
title('Box Plot of Variables');

window=16;
kkk=100;
hp=1;
beg1 = m(1) - kkk - window - hp;

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
end

% Error metrics calculation
MAE = mean(error, 1);
MSE = mean(error.^2, 1);
RMSE = sqrt(MSE);

% Calculate R-squared
SS_total = sum((Cn - mean(Cn)).^2);
SS_residual = sum(error.^2);
R_squared = 1 - (SS_residual ./ SS_total);

% Calculate MAPE
MAPE = mean(abs(error ./ Cn), 1) * 100;

% Calculate MAD
MAD = median(abs(error), 1);

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

