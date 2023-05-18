import numpy as np
import matplotlib.pyplot as plt

# Generate random data
data = np.random.randn(1000)

# Create a histogram from the data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(data, bins=30, alpha=0.75, color='blue', edgecolor='black')
plt.title('Original Histogram')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Normalize the histogram
plt.subplot(1, 2, 2)
weights = np.ones_like(data) / len(data)
plt.hist(data, bins=30, weights=weights, alpha=0.75, color='green', edgecolor='black')
plt.title('Normalized Histogram')
plt.xlabel('Values')
plt.ylabel('Probability')

plt.show()
