import numpy as np
import matplotlib.pyplot as plt

# --- Original Data (SAT-type Score vs. SAT Score) ---
Score = np.array([58, 48, 48, 41, 34, 43, 38, 53, 41, 60, 55, 44,  
                  43, 49, 47, 33, 47, 40, 46, 53, 40, 45, 39, 47,  
                  50, 53, 46, 53])
SAT = np.array([590, 590, 580, 490, 550, 580, 550, 700, 560, 690, 800, 600, 
                650, 580, 660, 590, 600, 540, 610, 580, 620, 600, 560, 560, 
                570, 630, 510, 620])
N = len(Score)

# 1. Calculate the Observed Correlation Coefficient (r)
r_obt = np.corrcoef(Score, SAT)[0, 1]
print(f"Observed Correlation (r): {r_obt:.6f}")

# 2. Perform Permutation Resampling
B = 10000 
r_random = np.empty(B)

# The core idea: under H0 (no correlation), any pairing of Score and SAT is equally likely.
# We fix one variable (e.g., Score) and randomly shuffle the other (SAT).
for i in range(B):
    # Randomly shuffle (permute) the SAT scores
    X_permuted = np.random.permutation(SAT)
    Y = Score # Keep Score fixed
    
    # Calculate the correlation for the permuted data
    r_random[i] = np.corrcoef(Y, X_permuted)[0, 1]

# 3. Calculate the P-value (One-sided: P(Stat_random >= Stat_observed))
# Testing for a positive correlation (r > 0)
pvalue = np.mean(np.concatenate(([r_obt], r_random)) >= r_obt)

print(f"Permutation P-value (B={B}): {pvalue:.6f}")

# 4. Plotting Results
plt.figure(figsize=(10, 6))
plt.hist(r_random, bins=30, density=True, color='gold', edgecolor='black', 
         label='Permutation Distribution (Correlation r)')
plt.axvline(r_obt, color='navy', linestyle='--', linewidth=2, label=f'Observed r ({r_obt:.4f})')
plt.title('Permutation Distribution for Correlation Coefficient')
plt.xlabel('Correlation Coefficient (r)')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.show()