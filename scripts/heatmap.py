# The puprose of that program is to show calendar hatmap of chosen feature form warsaw_data.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
df = pd.read_csv('trial10/warsaw_data.csv')
df['date'] = pd.to_datetime(df['date'])
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month
df['Day'] = df['date'].dt.day
pivot_df = df.pivot_table(values='pm2_5', index='Day', columns='Month', aggfunc=np.mean)
plt.figure(figsize=(10,10))
sns.heatmap(pivot_df, cmap='YlGnBu')
plt.title('Calendar Heatmap of PM2.5 values in 2022')
plt.figure(figsize=(10,10))
sns.heatmap(pivot_df, cmap='YlGnBu')
plt.title('Calendar Heatmap of PM2.5 values in 2022')
plt.savefig('heatmap.png')
plt.show()

