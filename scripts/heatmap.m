% Szymon Bartoszewicz
% The purpose of this script is to print heat xalendar for PM2.5.
data = readtable('warsaw_data.csv', 'VariableNamingRule', 'preserve');

data.date = datetime(data.date, 'InputFormat', 'yyyy-MM-dd');

data.Year = year(data.date);
data.Month = month(data.date);
data.Day = day(data.date);

% Calculate the mean PM2.5 values for each day of each month
meanPM25 = grpstats(data, {'Month', 'Day'}, 'mean', 'DataVars', 'pm2_5');

colorLimits = [min(meanPM25{:, 'mean_pm2_5'}(:)), max(meanPM25{:, 'mean_pm2_5'}(:))];

meanPM25 = unstack(meanPM25, 'mean_pm2_5', 'Month');

% Remove the first two columns
meanPM25(:, 1:2) = [];

figure;
h = heatmap(meanPM25.Variables, 'Colormap', parula);
h.Title = 'Calendar Heatmap of PM2.5 values in 2022';
h.XLabel = 'Month';
h.YLabel = 'Day';

h.ColorLimits = colorLimits;

% Display cell values with maximum 2 decimal places
h.CellLabelFormat = '%.2f';
