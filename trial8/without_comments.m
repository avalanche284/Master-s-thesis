
clear all
% Load data from CSV file
filename = 'warsaw_data.csv';
T = readtable(filename); % Read data from CSV file into a table
C = T{:, 2:12}; % Extract the columns without "date"
m = size(C);
summary(T)
column_names = T.Properties.VariableNames(2:12); % Extract the column names

figure;
boxplot(C, 'Labels', column_names);
title('Box Plot of Variables');

C_IQR = C;
num_vars = width(C_IQR);
for i = 1:num_vars
    column = C_IQR(:, i);
    [lower_bound, upper_bound] = iqr_bounds(column);
    C_IQR = C_IQR(column >= lower_bound & column <= upper_bound, :);
end
figure;
boxplot(C_IQR, 'Labels', column_names);
title('Box Plot of Variables after IQR');


for i = 1:size(C_IQR, 2)
    figure;
    histogram(C_IQR(:, i));
    xlabel(column_names{i});
    ylabel('Frequency');
    title(['(Before tranformation)Histogram of ' column_names{i}]);
 end


constant = 1;
C_normalized = C_IQR;
C_normalized(:, [1,2,5,6,11]) = log(C_IQR(:, [1,2,5,6,11]) + constant);
C_normalized(:, [11]) = sqrt(C_IQR(:, [11]) + constant);

C_normalized(:, [9]) = C_IQR(:, [9]).^3;
for i = 1:size(C_normalized, 2)
    figure;
    histogram(C_normalized(:, i));
    xlabel(column_names{i});
    ylabel('Frequency');
    title(['Histogram of transformed' column_names{i}]);
end

figure;
plotmatrix(C_IQR);
title('Scatter Plot Matrix of Variables');



std_dev = std(C_IQR);
fprintf('\nStandard Deviation for each feature:\n');
for i = 1:length(column_names)
    fprintf('%s: %.4f\n', column_names{i}, std_dev(i));
end


column_names = T.Properties.VariableNames(2:12);
figure
correlation_plot = corrplot(C_IQR, 'varNames', column_names);
title('Correlation plot');

partial_correlation_matrix = partialcorr(C_IQR);
partial_corr_plot(partial_correlation_matrix, column_names);

corr_matrix = corrcoef(C_IQR);
high_corr_threshold = 0.7; 
low_corr_threshold = 0.2;  
pm25_index = 1; % PM2.5 -- the first column
high_corr_with_pm25 = abs(corr_matrix(pm25_index, :)) >= high_corr_threshold;
low_corr_with_pm25 = abs(corr_matrix(pm25_index, :)) <= low_corr_threshold;

disp('Variables highly correlated with PM2.5:');
column_names(high_corr_with_pm25)

disp('Variables with low correlation with PM2.5:');
column_names(low_corr_with_pm25)


correlation_matrix = corrcoef(C_IQR);
figure;
heatmap(column_names, column_names, correlation_matrix);
title('Heatmap of Correlation Matrix');


T.date = datetime(T.date, 'InputFormat', 'yyyy-MM-dd');
start_summer = '2022-06-21';
end_summer = '2022-09-23';
filtered_summer = T(T.date >= datetime(start_summer, 'InputFormat', 'yyyy-MM-dd') & T.date <= datetime(end_summer, 'InputFormat', 'yyyy-MM-dd'), :);

if isempty(filtered_summer)
    disp('No data points found in the specified date range.');
else
    C_summer = filtered_summer{:, 2:end};

    constant = 1;
    C_summer_normalized = C_summer;
    C_summer_normalized(:, [1, 2, 4, 5, 6, 10]) = log(C_summer(:, [1, 2, 4, 5, 6, 10]) + constant);
    C_summer_normalized(:, [3, 11]) = sqrt(C_summer(:, [3, 11]) + constant);
    C_summer_normalized(:, [9]) = C_summer(:, [9]).^3;
end

start_winter = '2022-12-21';
end_winter = '2023-03-20';
filtered_winter = T((T.date >= datetime(start_winter, 'InputFormat', 'yyyy-MM-dd') & T.date <= datetime(end_winter, 'InputFormat', 'yyyy-MM-dd')), :);

if isempty(filtered_winter)
    disp('No data points found in the specified date range.');
else
    C_winter = filtered_winter{:, 2:end};

    constant = 1;
    C_winter_normalized = C_winter;
    C_winter_normalized(:, [1, 2, 4, 5, 6, 10]) = log(C_winter(:, [1, 2, 4, 5, 6, 10]) + constant);
    C_winter_normalized(:, [3, 11]) = sqrt(C_winter(:, [3, 11]) + constant);
    C_winter_normalized(:, [9]) = C_winter(:, [9]).^3;
end




selected_features =     [4, 5, 6]; % Selected feature indices from the original dataset
window =                7; % window size -- a number of data points taken from the past to make edicitons
kkk =                   52; % the number of maximum windows that can be used 
hp =                    1; % a prediction horizon
beg1 = m(1) - kkk - window - hp; % represents the starting index for the first window in the sliding window approach

m_transformed = size(C_normalized);
beg1 = m_transformed(1) - kkk - window - hp; % represents the starting index for the first window in the sliding window approach

for i = 1:kkk-1
    beg = beg1 + i;
    fin = beg + window;

    Cn(i, :) = C_IQR(fin + hp + 1, 1:2);
    C1 = C_IQR(beg:fin-1, selected_features); 
    y = C_IQR(beg + hp:fin + hp - 1, 1:2)';
    X = [ones(size(y, 2), 1), C1];
    A = X \ y'; % the Moore-Penrose pseudoinverse 
    Ym = X * A;
    dely = y - Ym';
    Xlast = [1, C_IQR(fin + hp + 1, selected_features)];
    ynext(i, :) = Xlast * A;
    error(i, :) = abs(ynext(i, :) - C_IQR(fin + hp + 1, 1:2));
    Ai(i, :, :) = A;

    Cn_transformed(i, :) = C_normalized(fin + hp + 1, 1:2);
    C1_transformed = C_normalized(beg:fin-1, selected_features); 
    y_transformed = C_normalized(beg + hp:fin + hp - 1, 1:2)';
    X_transformed = [ones(size(y_transformed, 2), 1), C1_transformed];
    A_transformed = X_transformed \ y_transformed'; % the Moore-Penrose pseudoinverse 
    Ym_transformed = X_transformed * A_transformed;
    dely_transformed = y_transformed - Ym_transformed';
    Xlast_transformed = [1, C_normalized(fin + hp + 1, selected_features)];
    ynext_transformed(i, :) = Xlast_transformed * A_transformed;
    error_transformed(i, :) = abs(ynext_transformed(i, :) - C_normalized(fin + hp + 1, 1:2));
    Ai_transformed(i, :, :) = A_transformed;
end


MAE = mean(error, 1);
MSE = mean(error.^2, 1);
RMSE = sqrt(MSE);
MAPE = mean(abs(error ./ Cn), 1) * 100;
MAD = median(abs(error), 1);
SS_total = sum((Cn - mean(Cn)).^2);
SS_residual = sum(error.^2);
R_squared = 1 - (SS_residual ./ SS_total);

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

figure
plot(10:40, ynext(10:40, 1), '-r')
hold on
plot(10:40, Cn(10:40, 1), '-g')
title('Model (red) and real (green) for PM2.5')

figure
plot(10:40, ynext(10:40, 2), '-r')
hold on
plot(10:40, Cn(10:40, 2), '-g')
title('Model (red) and real (green) for PM10')

ynext_transformed_inv = ynext_transformed;
ynext_transformed_inv(:, [1, 2]) = exp(ynext_transformed(:, [1, 2])) - constant;
MAE_transformed = mean(error_transformed, 1);
MSE_transformed = mean(error_transformed.^2, 1);
RMSE_transformed = sqrt(MSE_transformed);
MAPE_transformed = mean(abs(error_transformed ./ Cn_transformed), 1) * 100;
MAD_transformed = median(abs(error_transformed), 1);
SS_total_transformed = sum((Cn_transformed - mean(Cn_transformed)).^2);
SS_residual_transformed = sum(error_transformed.^2);
R_squared_transformed = 1 - (SS_residual_transformed ./ SS_total_transformed);
error_metrics_combined = [MAE; MAE_transformed; MSE; MSE_transformed; RMSE; RMSE_transformed; ...
    R_squared; R_squared_transformed; MAPE; MAPE_transformed; MAD; MAD_transformed];
error_metric_names_combined = {'MAE', 'MAE_norm', 'MSE', 'MSE_norm', 'RMSE', 'RMSE_norm', ...
    'R-squared', 'R-squared_norm', 'MAPE', 'MAPE_norm', 'MAD', 'MAD_norm'};

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
plot(10:40, ynext(10:40, 1), '-r', 'LineWidth', 2);
hold on;
plot(10:40, ynext_transformed_inv(10:40, 1), '--b', 'LineWidth', 2);
plot(10:40, Cn(10:40, 1), '-g', 'LineWidth', 2);
title('Model (red: non-normalized, blue: normalized) and real (green) for PM2.5');
legend('Non-normalized', 'Normalized', 'Real');
hold off;

subplot(2, 1, 2);
plot(10:40, ynext(10:40, 2), '-r', 'LineWidth', 2);
hold on;
plot(10:40, ynext_transformed_inv(10:40, 2), '--b', 'LineWidth', 2);
plot(10:40, Cn(10:40, 2), '-g', 'LineWidth', 2);
title('Model (red: non-normalized, blue: normalized) and real (green) for PM10');
legend('Non-normalized', 'Normalized', 'Real');
hold off;

figure;
bar(error_metrics_combined(:, 1));
set(gca, 'XTickLabel', error_metric_names_combined);
title('Error Metrics Comparison for PM2.5 Predictions (Normalized vs Non-normalized)');
ylabel('Error Value');
figure;
bar(error_metrics_combined(:, 2));
set(gca, 'XTickLabel', error_metric_names_combined);
title('Error Metrics Comparison for PM10 Predictions (Normalized vs Non-normalized)');
ylabel('Error Value');


error_metrics_table = array2table(error_metrics_combined, 'VariableNames', {'PM2_5', 'PM10'}, 'RowNames', error_metric_names_combined);
disp('Comparison Table of Error Metrics (Normalized vs Non-normalized):');
disp(error_metrics_table);

error_metrics_table = array2table(error_metrics_combined, 'VariableNames', {'PM2_5', 'PM10'}, 'RowNames', error_metric_names_combined);
error_metrics_table_transposed = cell2table(table2cell(error_metrics_table)', 'VariableNames', error_metric_names_combined, 'RowNames', {'PM2_5', 'PM10'});
writetable(error_metrics_table_transposed, 'err_comp.csv', 'WriteVariableNames', true, 'WriteRowNames', true);
disp('Transposed Comparison Table of Error Metrics (Normalized vs Non-normalized) has been saved to err_comp.csv');


% Create the "graphs" subfolder if it doesn't exist
if ~exist('graphs', 'dir')
    mkdir('graphs');
end

% Save each figure in the "graphs" subfolder as a JPG file
figHandles = findall(0, 'Type', 'figure');
for i = 1:length(figHandles)
    filename = sprintf('graphs/figure%d.jpg', i);
    saveas(figHandles(i), filename);
end

