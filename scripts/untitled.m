startDate = datetime(2022, 4, 2);
endDate = datetime(2023, 3, 31);

summerStart = datetime(2022, 6, 1);
summerEnd = datetime(2022, 8, 31);
winterStart1 = datetime(2022, 12, 1);
winterEnd1 = datetime(2022, 12, 31);
winterStart2 = datetime(2023, 1, 1);
winterEnd2 = datetime(2023, 2, 28);

wholeYearDays = days(endDate - startDate) + 1;
summerDays = days(summerEnd - summerStart) + 1;
winterDays = days(winterEnd1 - winterStart1) + 1 + days(winterEnd2 - winterStart2) + 1;
categories = {'Whole Year', 'Summer', 'Winter'};
numDays = [wholeYearDays, summerDays, winterDays];

figure;
bar(categorical(categories), numDays);
ylabel('Number of Days');
title('Dataset Split');
