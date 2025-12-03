import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp # Two-sample K-S test

# --- Data Re-use (chickwts: soybean vs. linseed) ---
X = np.array([219, 271, 258, 248, 240, 246, 254, 301, 280, 236, 234, 309, 253, 303]) # Soybean (n=14)
Y = np.array([148, 221, 203, 224, 250, 253, 269, 272, 237, 244, 253, 259]) # Linseed (n=12)
n1 = len(X) 
n2 = len(Y) 
Z = np.concatenate((X, Y))
N = len(Z) # 26

# 1. Calculate the Observed K-S Statistic (D)
# ks_2samp returns D and p-value. We only need D.
D_obs = ks_2samp(X, Y).statistic
print(f"Observed K-S Statistic (D): {D_obs:.4f}")

# 2. Perform Permutation Resampling
B = 10000 
D_stats = np.empty(B)

for i in range(B):
    # Randomly choose n1 indices for the new 'X' group
    k = np.random.choice(N, size=n1, replace=False)
    
    x1 = Z[k]
    
    # The remaining indices form the new 'Y' group
    y1_mask = np.ones(N, dtype=bool)
    y1_mask[k] = False
    y1 = Z[y1_mask]
    
    # Calculate the K-S statistic for the permuted data
    D_stats[i] = ks_2samp(x1, y1).statistic

# 3. Calculate the P-value (One-sided: P(Stat_random >= Stat_observed))
# Large values of D support the alternative (different distributions)
pvalue = np.mean(np.concatenate(([D_obs], D_stats)) >= D_obs)

print(f"Permutation P-value (B={B}): {pvalue:.6f}")

# 4. Plotting Results
plt.figure(figsize=(10, 6))
plt.hist(D_stats, bins=30, density=True, color='mediumseagreen', edgecolor='black', 
         label='Permutation Distribution (K-S D)')
plt.axvline(D_obs, color='darkviolet', linestyle='--', linewidth=2, label=f'Observed D Stat ({D_obs:.2f})')
plt.title('Permutation Distribution for Kolmogorov-Smirnov Statistic')
plt.xlabel('K-S Statistic (D)')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.show()