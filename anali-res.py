# The sole purpose of this script is to analize the Results Table of my masters' research.

import pandas as pd

# Read the CSV file
data = pd.read_csv('results.csv', delimiter=',') # If your CSV uses commas, replace '\t' with ','

# Display a summary of the data
print(data.describe())

# Find the best (minimum) MAE and corresponding row
best_mae = data.loc[data['MAE'].idxmin()]
print("\nBest MAE:")
print(best_mae)

# Find the best (minimum) RMSE and corresponding row
best_rmse = data.loc[data['RMSE'].idxmin()]
print("\nBest RMSE:")
print(best_rmse)

# Find the best (maximum) R-squared and corresponding row
best_r_squared = data.loc[data['R-squared'].idxmax()]
print("\nBest R-squared:")
print(best_r_squared)
