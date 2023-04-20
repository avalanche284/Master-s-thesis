% trial10summer.m --> Based on trial9.m
% The file contain computation do predict PM2.5 and PM10 in the summer
% period by use of the Mutlivariate regression method, sliding window
% approach and Moore-Penrose pseudoinverse.
% 

% Normalization
% This script has both versions: with normalizaiotn and without it.
% It is intended to be run concurrently so that error measurements and
% plots in the end show differences in prediction accuracy. However, if you
% prefer cleaner output meaning -- without unnecessary graphs -- just
% comment/uncomment what is needed. Part of the script that relate to
% normalization are marked as below.
% ### Normalization: This block of code applies normalization to the dataset

% Additional files
% In the script are used functions which are defined by
% their names in the path as "function_name.m".

% Comments
% Some parts of the script has two comment sings. It was only done to limit
% the number of plots drawn by MATLAB. Uncomment if you need those
% (CTRL + SHIFT + R on PC, COMMAND + SHIFT + / on Mac)
%% karkat (C) AW 2021
%% Szymon Bartoszewicz WSB Merito University in Gdansk, 2023
% 
% The purpose of this script is to forecast PM2.5 and PM10 concurrently
% based on predictors: pm2_5 pm10 o3 no2 so2 co temp pressure humidity wind_speed clouds.
% Pollutant concentration in μg/m3. Units:temp -- kelvin, pressure -- hPa,
% humidity -- %, wind_speed -- meter/sec, clouds -- %.
% The dataset comes from OpenWeather® --> apd21_air_pollution.py
% & apd51_weatehr_data.py
% Datasets were combined to get daily averages of pollutants and weather
% conditions i.e., temp	pressure	humidity	wind_speed	clouds	pm2_5
% pm10	o3	no2	so2	co.
% The script uses the sliding window approach and Moore-Penrose
% pseudoinverse. Additionally, it contains error measures such as R-squared
% and MAPE RMSE, MSE, MAD.

clear all
filename = 'warsaw_data.csv';
T = readtable(filename); % Read data from CSV file into a table
C = T{:, 2:12}; % Extract the columns without "date"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1.1. Summer period
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating a dataset from the existing one containing data only from summer
% period.

T.date = datetime(T.date, 'InputFormat', 'yyyy-MM-dd');
start_summer = '2022-06-21';
end_summer = '2022-09-23';
filtered_summer = T(T.date >= datetime(start_summer, 'InputFormat', 'yyyy-MM-dd') & T.date <= datetime(end_summer, 'InputFormat', 'yyyy-MM-dd'), :);
% Check if any data points match the specified date range
if isempty(filtered_summer)
    disp('No data points found in the specified date range.');
else
    % Remove the date column from the filtered_summer
    C_summer = filtered_summer{:, 2:end};
    C_IQR_summer = C_summer;
    num_vars_summer = width(C_IQR_summer);
    % IQR on summer period
    for i = 1:num_vars_summer
        column_summer = C_IQR_summer(:, i);
        [lower_bound_summer, upper_bound_summer] = iqr_bounds(column_summer);
        C_IQR_summer = C_IQR_summer(column_summer >= lower_bound_summer & column_summer <= upper_bound_summer, :);
    end
    % Apply the normalization to C_summer
    constant = 1;
    C_summer_normalized = C_IQR_summer;
    C_summer_normalized(:, [1, 2, 4, 5, 6, 10]) = log(C_IQR_summer(:, [1, 2, 4, 5, 6, 10]) + constant);
    C_summer_normalized(:, [3, 11]) = sqrt(C_IQR_summer(:, [3, 11]) + constant);
    C_summer_normalized(:, [9]) = C_IQR_summer(:, [9]).^3;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% feature order:
% +-------+------+----+-----+-----+----+------+----------+----------+------------+---------+
% |   1   |  2   | 3  |  4  |  5  | 6  |  7   |    8     |    9     |     10     |   11    |
% +-------+------+----+-----+-----+----+------+----------+----------+------------+---------+
% | pm2_5 | pm10 | o3 | no2 | so2 | co | temp | pressure | humidity | wind_speed | clouds  |
% +-------+------+----+-----+-----+----+------+----------+----------+------------+---------+

selected_features =     [4, 5, 6]; % Selected feature indices from the original dataset
window =                7; % window size -- a number of data points taken from the past to make edicitons
kkk =                   13; % the number of maximum windows that can be used 
hp =                    1; % a prediction horizon
 
m_summer = size(C_IQR_summer);
beg1 = m_summer(1) - kkk - window - hp;

for i = 1:kkk-1
    beg = beg1 + i;
    fin = beg + window;
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Summer
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Cn(i, :) = C_IQR_summer(fin + hp + 1, 1:2);    
    C1 = C_IQR_summer(beg:fin-1, selected_features); 
    y = C_IQR_summer(beg + hp:fin + hp - 1, 1:2)';
    X = [ones(size(y, 2), 1), C1];
    A = X \ y'; % the Moore-Penrose pseudoinverse 
    Ym = X * A;
    dely = y - Ym';
    Xlast = [1, C_IQR_summer(fin + hp + 1, selected_features)];
    ynext(i, :) = Xlast * A;
    error(i, :) = abs(ynext(i, :) - C_IQR_summer(fin + hp + 1, 1:2));
    Ai(i, :, :) = A;
    % ### Normalization:
    Cn_transformed(i, :) = C_summer_normalized(fin + hp + 1, 1:2);
    C1_transformed = C_summer_normalized(beg:fin-1, selected_features); 
    y_transformed = C_summer_normalized(beg + hp:fin + hp - 1, 1:2)';
    X_transformed = [ones(size(y_transformed, 2), 1), C1_transformed];
    A_transformed = X_transformed \ y_transformed'; % the Moore-Penrose pseudoinverse 
    Ym_transformed = X_transformed * A_transformed;
    dely_transformed = y_transformed - Ym_transformed';
    Xlast_transformed = [1, C_summer_normalized(fin + hp + 1, selected_features)];
    ynext_transformed(i, :) = Xlast_transformed * A_transformed;
    error_transformed(i, :) = abs(ynext_transformed(i, :) - C_summer_normalized(fin + hp + 1, 1:2));
    Ai_transformed(i, :, :) = A_transformed;


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. Prediction errors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
fprintf('\nPM2.5 Prediction Metrics:\n');
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

% the range -- change value below e.g., @1@0:@4@0, @1:@1@1, 1@:2@.
% Be careful with 1:2 -- "find and replace all" will make changes also in
% the main loop of the code above -- you don't want to change those.
figure
plot(1:12, ynext(1:12, 1), '-r')
hold on
plot(1:12, Cn(1:12, 1), '-g')
title('Model (red) and real (green) for PM2.5')

figure
plot(1:12, ynext(1:12, 2), '-r')
hold on
plot(1:12, Cn(1:12, 2), '-g')
title('Model (red) and real (green) for PM10')

% ### Normalization
% Inverse transformations
ynext_transformed_inv = ynext_transformed;
% Inverse log transformation for PM2.5 and PM10
ynext_transformed_inv(:, [1, 2]) = exp(ynext_transformed(:, [1, 2])) - constant;
% Error metrics calculation for normalized version
MAE_transformed = mean(error_transformed, 1);
MSE_transformed = mean(error_transformed.^2, 1);
RMSE_transformed = sqrt(MSE_transformed);
MAPE_transformed = mean(abs(error_transformed ./ Cn_transformed), 1) * 100;
MAD_transformed = median(abs(error_transformed), 1);
% R-squared for normalized version
SS_total_transformed = sum((Cn_transformed - mean(Cn_transformed)).^2);
SS_residual_transformed = sum(error_transformed.^2);
R_squared_transformed = 1 - (SS_residual_transformed ./ SS_total_transformed);
% Combine error metrics from both normalized and non-normalized versions
error_metrics_combined = [MAE; MAE_transformed; MSE; MSE_transformed; RMSE; RMSE_transformed; ...
    R_squared; R_squared_transformed; MAPE; MAPE_transformed; MAD; MAD_transformed];
error_metric_names_combined = {'MAE', 'MAE_norm', 'MSE', 'MSE_norm', 'RMSE', 'RMSE_norm', ...
    'R-squared', 'R-squared_norm', 'MAPE', 'MAPE_norm', 'MAD', 'MAD_norm'};

% Display error metrics and R-squared
fprintf('\nPM2.5 Prediction Metrics:\n');
fprintf('MAE_transformed: %.4f\n', MAE_transformed(1));
fprintf('MSE_transformed: %.4f\n', MSE_transformed(1));
fprintf('RMSE_transformed: %.4f\n', RMSE_transformed(1));
fprintf('R-squared: %.4f\n', R_squared(1));
fprintf('MAPE_transformed: %.4f%%\n', MAPE_transformed(1));
fprintf('MAD_transformed: %.4f\n', MAD_transformed(1));

fprintf('\nPM10 Prediction Metrics:\n');
fprintf('MAE_transformed: %.4f\n', MAE_transformed(2));
fprintf('MSE_transformed: %.4f\n', MSE_transformed(2));
fprintf('RMSE_transformed: %.4f\n', RMSE_transformed(2));
fprintf('R-squared: %.4f\n', R_squared(2));
fprintf('MAPE_transformed: %.4f%%\n', MAPE_transformed(2));
fprintf('MAD_transformed: %.4f\n', MAD_transformed(2));

figure;
subplot(2, 1, 1);
plot(1:12, ynext(1:12, 1), '-r', 'LineWidth', 2);
hold on;
plot(1:12, ynext_transformed_inv(1:12, 1), '--b', 'LineWidth', 2);
plot(1:12, Cn(1:12, 1), '-g', 'LineWidth', 2);
title('Model and real for PM2.5');
legend('Non-normalized', 'Normalized', 'Real');
hold off;

subplot(2, 1, 2);
plot(1:12, ynext(1:12, 2), '-r', 'LineWidth', 2);
hold on;
plot(1:12, ynext_transformed_inv(1:12, 2), '--b', 'LineWidth', 2);
plot(1:12, Cn(1:12, 2), '-g', 'LineWidth', 2);
title('Model and real for PM10');
legend('Non-normalized', 'Normalized', 'Real');
hold off;

% Bar Plot for Error Metrics Comparison (PM2.5)
figure;
bar(error_metrics_combined(:, 1));
set(gca, 'XTickLabel', error_metric_names_combined);
title('Error Metrics Comparison for PM2.5');
ylabel('Error Value');
% Bar Plot for Error Metrics Comparison (PM10)
figure;
bar(error_metrics_combined(:, 2));
set(gca, 'XTickLabel', error_metric_names_combined);
title('Error Metrics Comparison for PM10');
ylabel('Error Value');

% % Bar Plot for Error Metrices
% % Not sure whther it gives a useful piece of information.
% error_metrics = [MAE; MSE; RMSE; R_squared; MAPE; MAD];
% error_metric_names = {'MAE', 'MSE', 'RMSE', 'R-squared', 'MAPE', 'MAD'};
% figure;
% bar(error_metrics);
% set(gca, 'XTickLabel', error_metric_names);
% legend('PM2.5', 'PM10');
% title('Error Metrics for PM2.5 and PM10 Predictions');

% Saving plots to a folder. Change its name depending on the number of
% study you conduct.
if ~exist('testsum2', 'dir')
    mkdir('testsum2');
end
% Save each figure in the "plots" subfolder as a JPG file
figHandles = findall(0, 'Type', 'figure');
for i = 1:length(figHandles)    
    filename = sprintf('testsum2/figure%d.jpg', i);
    saveas(figHandles(i), filename);
end
% Create a table for the comparison of error metrics
error_metrics_table = array2table(error_metrics_combined, 'VariableNames', {'PM2_5', 'PM10'}, 'RowNames', error_metric_names_combined);
disp('Comparison Table of Error Metrics:');
disp(error_metrics_table);
% Transpose the table
error_metrics_table_transposed = cell2table(table2cell(error_metrics_table)', 'VariableNames', error_metric_names_combined, 'RowNames', {'PM2_5', 'PM10'});
writetable(error_metrics_table_transposed, 'testsum2/sum_error_comparison.csv', 'WriteVariableNames', true, 'WriteRowNames', true);
disp('Transposed Comparison Table of Error Metrics has been saved to testsum2/sum_error_comparison.csv');

