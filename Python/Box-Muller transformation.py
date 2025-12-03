import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm

def bmnormal(n, mu=0.0, sd=1.0):
    """
    Simulates n draws from a Normal(mu, sd) distribution using the 
    Box-Muller transformation.

    Parameters:
    n (int): Number of draws to simulate.
    mu (float): Mean of the target Normal distribution.
    sd (float): Standard deviation of the target Normal distribution.

    Returns:
    numpy.ndarray: Array of n normally distributed random variables.
    """
    if n <= 0 or sd <= 0:
        raise ValueError("n must be positive and sd must be positive.")
    
    # We need ceil(n / 2) pairs of uniform random numbers to generate n draws.
    m = int(np.ceil(n / 2))
    
    # 1. Generate two sets of independent uniform variables U1 and U2
    U1 = np.random.uniform(0.0, 1.0, m)
    U2 = np.random.uniform(0.0, 1.0, m)
    
    # 2. Apply the Box-Muller transformation
    R = np.sqrt(-2.0 * np.log(U1))
    Theta = 2.0 * np.pi * U2
    
    # Z1 and Z2 are independent standard normal deviates (N(0, 1))
    Z1 = R * np.cos(Theta)
    Z2 = R * np.sin(Theta)
    
    # 3. Combine and truncate the results to n draws
    Z = np.concatenate((Z1, Z2))[:n]
    
    # 4. Scale and shift to the target Normal(mu, sd) distribution
    X = mu + sd * Z
    
    return X

# --- Simulation and Verification ---
N_DRAWS = 2000
MU = 10.0
SD = 3.0

# Generate samples
np.random.seed(42)
sim_samples = bmnormal(N_DRAWS, MU, SD)

print(f"--- Simulation Results (N={N_DRAWS}, μ={MU}, σ={SD}) ---")
print(f"Sample Mean: {np.mean(sim_samples):.4f}")
print(f"Sample Std Dev: {np.std(sim_samples, ddof=1):.4f}")

## Plotting to be "convinced" 🖼️
# 1. Histogram of Samples vs. True PDF
x_range = np.linspace(MU - 4*SD, MU + 4*SD, 100)
true_pdf = norm.pdf(x_range, loc=MU, scale=SD)

plt.figure(figsize=(10, 6))
plt.hist(sim_samples, bins=30, density=True, alpha=0.6, color='skyblue', 
         edgecolor='black', label='Box-Muller Samples')
plt.plot(x_range, true_pdf, 'r-', linewidth=2, label=f'True Normal(μ={MU}, σ={SD}) PDF')
plt.title('Box-Muller Generated Samples vs. True Normal Distribution')
plt.xlabel('Value (X)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

## Statistical Test 🧪
# 2. Kolmogorov-Smirnov (K-S) Test for Normality
# H0: The sample data comes from the specified theoretical distribution (Normal(mu, sd)).
# A high p-value (> 0.05) supports the null hypothesis.
ks_statistic, p_value = kstest(sim_samples, 'norm', args=(MU, SD))

print("\n--- Statistical Test (Kolmogorov-Smirnov) ---")
print(f"K-S Statistic: {ks_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value > 0.05:
    print("Conclusion: Since p > 0.05, we do not reject H0. The samples appear to be Normally distributed.")
else:
    print("Conclusion: Since p <= 0.05, we reject H0. The samples are unlikely to be Normally distributed.")