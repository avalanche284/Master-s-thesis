% partial_corr_plot.m
function partial_corr_plot(partial_correlation_matrix, varNames)
    numVars = size(partial_correlation_matrix, 1);
    imagesc(partial_correlation_matrix, [-1 1]);
    colormap(jet);
    colorbar;
    set(gca, 'XTick', 1:numVars, 'XTickLabel', varNames, 'YTick', 1:numVars, 'YTickLabel', varNames);
    title('Partial Correlation Graph');
end
