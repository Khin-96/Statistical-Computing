import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# --- Original Data (DRP Scores) ---
T = np.array([24, 43, 58, 71, 61, 44, 67, 49, 59, 52, 62, 54, 46, 43, 57, 
              43, 57, 56, 53, 49, 33])
C = np.array([42, 43, 55, 26, 33, 41, 19, 54, 46, 10, 17, 60, 37, 42, 55, 
              28, 62, 53, 37, 42, 20, 48, 85])
n1 = len(T)
n2 = len(C)
Z = np.concatenate((T, C)) # Pooled data
N = len(Z)

# 1. Calculate the Observed Statistic
obs_stat = np.mean(T) - np.mean(C)
print(f"Observed Difference in Means: {obs_stat:.4f}")

# 2. Perform Permutation Resampling
B = 10000 # Increased B for better P-value estimation
new_stats = np.empty(B)

# The core idea: under H0 (no difference), the group labels are interchangeable.
for i in range(B):
    # Randomly choose n1 indices for the new 'Treatment' group
    idx = np.random.choice(N, size=n1, replace=False)
    
    newT = Z[idx]
    # The remaining indices form the new 'Control' group
    # Note: np.delete is clean but can be slow; list comprehension is an alternative
    newC_mask = np.ones(N, dtype=bool)
    newC_mask[idx] = False
    newC = Z[newC_mask]
    
    new_stats[i] = np.mean(newT) - np.mean(newC)

# 3. Calculate the P-value (One-sided: P(Stat_random >= Stat_observed))
# Include the observed statistic in the null distribution for a common definition
pvalue = np.mean(np.concatenate(([obs_stat], new_stats)) >= obs_stat)

print(f"Permutation P-value (B={B}): {pvalue:.6f}")

# 4. Plotting Results
plt.figure(figsize=(10, 6))
plt.hist(new_stats, bins=30, density=True, color='skyblue', edgecolor='black', 
         label='Permutation Distribution')
plt.axvline(obs_stat, color='red', linestyle='--', linewidth=2, label=f'Observed Stat ({obs_stat:.2f})')
plt.title('Permutation Distribution for Difference in Means')
plt.xlabel(r'$\bar{T} - \bar{C}$')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.show()