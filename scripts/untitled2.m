% boxplot analysis
filename = 'warsaw_data.csv';
data = readmatrix(filename);

% Calculate the summary statistics for each feature
for column_number = 2:size(data, 2)
    % Median
    median_value = median(data(:, column_number));

    % Quartile 1 (Q1)
    q1 = quantile(data(:, column_number), 0.25);

    % Quartile 3 (Q3)
    q3 = quantile(data(:, column_number), 0.75);

    % Minimum
    min_value = min(data(:, column_number));

    % Maximum
    max_value = max(data(:, column_number));

    % Outliers
    iqr = q3 - q1;
    lower_bound = q1 - 1.5 * iqr;
    upper_bound = q3 + 1.5 * iqr;
    outliers = data((data(:, column_number) < lower_bound) | (data(:, column_number) > upper_bound), column_number);

    fprintf('Feature %d:\n', column_number - 1);
    fprintf('Median: %f\n', median_value);
    fprintf('Q1: %f\n', q1);
    fprintf('Q3: %f\n', q3);
    fprintf('Min: %f\n', min_value);
    fprintf('Max: %f\n', max_value);
    fprintf('Outliers: ');
    fprintf('%f ', outliers);
    fprintf('\n\n');
end
