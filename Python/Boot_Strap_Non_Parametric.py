import numpy as np
import scipy.stats as st

# --- Sample Data (mtcars data for simplicity) ---
data = np.array([21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 
                 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 
                 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 
                 15.0, 21.4])
N = len(data)

# --- Statistic Function ---
# Calculate the median of the data (or any other statistic like R-squared)
def calculate_statistic(sample):
    return np.median(sample)

# 1. Calculate the Observed Statistic
obs_stat = calculate_statistic(data)

# 2. Perform Bootstrap Resampling
B = 1000 # Number of bootstrap replicates
bootstrap_stats = np.empty(B)

np.random.seed(42)

for i in range(B):
    # Resample with replacement from the original data
    bootstrap_sample = np.random.choice(data, size=N, replace=True)
    bootstrap_stats[i] = calculate_statistic(bootstrap_sample)

# 3. Calculate Confidence Interval (Percentile Method)
confidence_level = 0.95
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100

ci_percentile = np.percentile(bootstrap_stats, [lower_percentile, upper_percentile])

# 4. Calculate Bias and Standard Error
bias = np.mean(bootstrap_stats) - obs_stat
std_error = np.std(bootstrap_stats, ddof=1)

print(f"Observed Statistic (Median): {obs_stat:.4f}")
print("--- Bootstrap Statistics ---")
print(f"Bias: {bias:.4f}")
print(f"Standard Error: {std_error:.4f}")
print(f"{confidence_level*100:.0f}% Percentile CI: {ci_percentile}")

# Note: The 'bca' method shown in the R examples (Bias-Corrected and Accelerated) 
# is more complex and typically requires specialized libraries like `arch` or `scikits-bootstrap` 
# in Python, but the percentile method provides the fundamental CI concept.