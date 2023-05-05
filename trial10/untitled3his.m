% Read the CSV file
filename = 'warsaw_data.csv';
data = readtable(filename);

% Remove the date column since it's not a numerical feature
data(:,1) = [];

% Create a structure to store the results
stats = struct();

% Loop over all the features
for i = 1 : width(data)
    feature_name = data.Properties.VariableNames{i};
    feature_data = data{:, i};

    % Calculate statistics
    [num_bins, edges] = histcounts(feature_data);
    mean_value = mean(feature_data);
    median_value = median(feature_data);
    mode_value = mode(feature_data);
    range_value = range(feature_data);
    q3_value = quantile(feature_data, 0.75);
    q1_value = quantile(feature_data, 0.25);
    iqr_value = q3_value - q1_value;
    var_value = var(feature_data);
    std_value = std(feature_data);

    % Store the results in the structure
    stats.(feature_name) = struct('raw_data_bins', num_bins, ...
                                   'bin_ranges', edges(1:end-1), ...
                                   'frequency_counts', num_bins, ...
                                   'mean', mean_value, ...
                                   'median', median_value, ...
                                   'mode', mode_value, ...
                                   'range', range_value, ...
                                   'interquartile_range', iqr_value, ...
                                   'variance', var_value, ...
                                   'standard_deviation', std_value);
end

% Display the statistics
%disp(stats);

% Loop over all the features in the stats structure
for feature_name = fieldnames(stats)'
    fprintf('Feature: %s\n', feature_name{1});
    feature_stats = stats.(feature_name{1});
    disp(feature_stats);
    fprintf('\n');
end

