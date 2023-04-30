    % exploration.m
% This file contains computations required to perform desription part of
% the dataset. It is exploraton of the dataset.

clear all
filename = 'warsaw_data.csv';
T = readtable(filename); % Read data from CSV file into a table
C = T{:, 2:12}; % Extract the columns without "date"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Descriptive statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summary function
m = size(C);
summary(T)
column_names = T.Properties.VariableNames(2:12); % Extract the column names
% from the table T

% Box Plots
% A boxplot is a standardized way of displaying the distribution of data
% based on a five number summary (“minimum”, first quartile [Q1], median,
% third quartile [Q3] and “maximum”). It can tell you about your outliers
% and what their values are.
figure;
boxplot(C, 'Labels', column_names);
title('Box Plot of Variables');
% After analysis of boxplots I state that removing outliers is necessary.
% The method that will be used here is IQR method.
% Apply the IQR method to all variables
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

% Histograms
% It's a diagram consisting of rectangles whose area is proportional to the
% frequency of a variable and whose width is equal to the class interval.
for i = 1:size(C_IQR, 2)
    figure;
    histogram(C_IQR(:, i));
    xlabel(column_names{i});
    ylabel('Frequency');
    title(['Histogram of ' column_names{i}]);
 end

figure;
plotmatrix(C_IQR);
title('Scatter Plot Matrix before normalization');

% ### Normalization
% After examination of above histogram charts, I state that:
% +-------+------+-----+-----+------+------+------+----------+----------+------------+---------+
% | 1     | 2    | 3   | 4   | 5    | 6    | 7    | 8        | 9        | 10         | 11      |
% +-------+------+-----+-----+------+------+------+----------+----------+------------+---------+
% | pm2_5 | pm10 | o3  | no2 | so2  | co   | temp | pressure | humidity | wind_speed | clouds  |
% | r-sk  | r-sk | nor | nor | r-sk | r-sk | nor  | nor      | l-sk     | nor        | r-sk    |
% +-------+------+-----+-----+------+------+------+----------+----------+------------+---------+
% where:
% r-sk -- right-skewed
% l-sk -- left-skewed
% nor -- normla distribution.

% Having the above in mind, cube and log transformations will be applied
% sequentially.
% Addition a small constant to handle zeros in the data (if any) for log transformation
constant = 1;
% Initialization the transformed dataset with the original data
C_normalized = C_IQR;
% Log transformation to right-skewed features
C_normalized(:, [1,2,5,6,11]) = log(C_IQR(:, [1,2,5,6,11]) + constant);
% Previous tests shew clouds' data durability to log tranformation. That is
% why square root transformation is utilized in this case.
C_normalized(:, [11]) = sqrt(C_IQR(:, [11]) + constant);

% Cube transformation to left-skewed features
C_normalized(:, [9]) = C_IQR(:, [9]).^3;
for i = 1:size(C_normalized, 2)
    figure;
    histogram(C_normalized(:, i));
    xlabel(column_names{i});
    ylabel('Frequency');
    title(['Histogram of ' column_names{i} ' after normalization']);
end
% ### Normalization: End

% Scatter plot matrix
% A scatter plot matrix is a grid (or matrix) of scatter plots used to
% visualize bivariate relationships between combinations of variables. Each
% scatter plot in the matrix visualizes the relationship between a pair
% of variables.
figure;
plotmatrix(C_normalized);
title('Scatter Plot Matrix after normalization');


% Standard deviation
% The standard deviation helps to understand how much the individual data
% points differ from the mean of the dataset.
std_dev = std(C_IQR);
fprintf('\nStandard Deviation for each feature:\n');
for i = 1:length(column_names)
    fprintf('%s: %.4f\n', column_names{i}, std_dev(i));
end

% Correlation plots
% Predictors should be chosen such that they have high correlaton with the
% response variable and low correlation between each other (to avoid
% redundancy).
column_names = T.Properties.VariableNames(2:12);
figure
correlation_plot = corrplot(C_IQR, 'varNames', column_names);
title('Correlation plot');

% Calculate the partial correlation coefficients (controlling for all other variables)
partial_correlation_matrix = partialcorr(C_IQR);
partial_corr_plot(partial_correlation_matrix, column_names);

% Show high and low correlations
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

% Heatmap
% Correlation heatmaps are a type of plot that visualize the strength of
% relationships between numerical variables.
correlation_matrix = corrcoef(C_IQR);
figure;
heatmap(column_names, column_names, correlation_matrix);
title('Heatmap of Correlation Matrix');


% Saving plots to a folder.
if ~exist('Plots_exploring_the_dataset', 'dir')
    mkdir('Plots_exploring_the_dataset');
end
% Save each figure in the "plots" subfolder as a JPG file
figHandles = findall(0, 'Type', 'figure');
for i = 1:length(figHandles)
    filename = sprintf('Plots_exploring_the_dataset/figure%d.jpg', i);
    saveas(figHandles(i), filename);
end
