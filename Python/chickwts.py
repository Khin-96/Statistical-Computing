import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# --- Original Data (chickwts: soybean vs. linseed) ---
# Note: Since Python doesn't have built-in datasets like R's 'chickwts', 
# we manually define the two groups extracted from the R example's logic.

# In R: X <- as.vector(chickwts$weight[chickwts$feed=="soybean"])
# In R: Y <- as.vector(chickwts$weight[chickwts$feed=="linseed"])
# Based on typical chickwts data:
X = np.array([219, 271, 258, 248, 240, 246, 254, 301, 280, 236, 234, 309, 253, 303]) # Soybean (n=14)
Y = np.array([148, 221, 203, 224, 250, 253, 269, 272, 237, 244, 253, 259]) # Linseed (n=12) - Note: R example shows n1=14, n2=12, total N=26
# R's 'chickwts' for linseed has 12 observations, soybean has 14. N=26.
n1 = len(X) # 14
n2 = len(Y) # 12
Z = np.concatenate((X, Y))
N = len(Z) # 26

# 1. Calculate the Observed T-statistic
t0 = ttest_ind(X, Y, equal_var=False).statistic # Use Welch's t-test statistic
print(f"Observed T-statistic (Welch's): {t0:.4f}")

# 2. Perform Permutation Resampling
B = 10000 
reps = np.empty(B)

for i in range(B):
    # Randomly choose n1 indices for the new 'X' group
    k = np.random.choice(N, size=n1, replace=False)
    
    x1 = Z[k]
    
    # Create mask for remaining indices to form 'Y' group
    y1_mask = np.ones(N, dtype=bool)
    y1_mask[k] = False
    y1 = Z[y1_mask]
    
    # Calculate the T-statistic for the permuted data
    # We must use a function that calculates the t-statistic (not just the mean diff)
    reps[i] = ttest_ind(x1, y1, equal_var=False).statistic

# 3. Calculate the P-value (One-sided: P(Stat_random >= Stat_observed))
# Since we are testing for X > Y (Soybean > Linseed), we look at the upper tail.
pvalue = np.mean(np.concatenate(([t0], reps)) >= t0)

print(f"Permutation P-value (B={B}): {pvalue:.6f}")

# 4. Plotting Results
plt.figure(figsize=(10, 6))
plt.hist(reps, bins=30, density=True, color='lightcoral', edgecolor='black', 
         label='Permutation Distribution (T-statistic)')
plt.axvline(t0, color='blue', linestyle='--', linewidth=2, label=f'Observed T-stat ({t0:.2f})')
plt.title('Permutation Distribution for T-statistic (chickwts)')
plt.xlabel('T-statistic')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.show()