import pandas as pd

# Read the CSV file
data = pd.read_csv('results kkk3 removed.csv', delimiter=',') # If your CSV uses commas, replace '\t' with ','

# Display a summary of the data
print(data.describe())

# Create a function to find and print the best value for each indicator
def find_best(indicator):
    if indicator in ['R-squared', 'R-squared_norm']:
        best_value = data.loc[data[indicator].idxmax()]
    else:
        best_value = data.loc[data[indicator].idxmin()]
    print(f"\nBest {indicator}:")
    print(best_value)

# Find the best values for all the indicators
indicators = ['MAE', 'MAE_norm', 'RMSE', 'RMSE_norm', 'R-squared', 'R-squared_norm', 'MAPE', 'MAPE_norm']

for indicator in indicators:
    find_best(indicator)
