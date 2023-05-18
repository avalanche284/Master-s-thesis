% iqr_bounds function
function [lower_bound, upper_bound] = iqr_bounds(column)
    Q1 = quantile(column, 0.25);
    Q3 = quantile(column, 0.75);
    IQR = Q3 - Q1;
    lower_bound = Q1 - 1.5 * IQR;
    upper_bound = Q3 + 1.5 * IQR;
end