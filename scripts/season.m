data = readtable('warsaw_data.csv');
data.date = datetime(data.date, 'InputFormat', 'yyyy-MM-dd');
windowSize = 12; % Adjust the window size according to the seasonality period.
smoothed_pm2_5 = movmean(data.pm2_5, windowSize);
figure;
plot(data.date, data.pm2_5, 'DisplayName', 'Original Data');
hold on;
plot(data.date, smoothed_pm2_5, 'DisplayName', 'Smoothed Data');
xlabel('Date');
ylabel('PM2.5');
legend;
hold off;
