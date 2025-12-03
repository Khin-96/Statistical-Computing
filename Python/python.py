################################################
# COMBINED PYTHON SCRIPT  
# Monte Carlo • Bootstrap • Simulation • Optimization  
# Works with NumPy + SciPy  
################################################

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

################################################
# 1. BOOTSTRAP RESAMPLING FUNCTION
################################################

def bootstrap_resample(arr):
    """Return bootstrap resample of same length."""
    return np.random.choice(arr, size=len(arr), replace=True)

# Example: 5 bootstrap samples of 6–10
bootstrap_examples = np.column_stack([
    bootstrap_resample(np.arange(6, 11)) for _ in range(5)
])

print("Bootstrap sample examples (6–10 repeated 5 times):")
print(bootstrap_examples)
print("\n")


################################################
# 2. BOOTSTRAP TWO-SAMPLE DIFFERENCE IN MEANS
################################################

def diff_in_means(x, y):
    """Difference in sample means."""
    return np.mean(x) - np.mean(y)

# Example data (YOU CAN REPLACE THESE)
x = np.random.normal(loc=5, scale=2, size=20)
y = np.random.normal(loc=6, scale=2, size=25)

B = 1000
boot_stats = []

for _ in range(B):
    x_star = np.random.choice(x, size=len(x), replace=True)
    y_star = np.random.choice(y, size=len(y), replace=True)
    boot_stats.append(diff_in_means(x_star, y_star))

boot_ci = np.quantile(boot_stats, [0.025, 0.975])

print("Bootstrap 95% CI for difference in means:")
print(boot_ci)
print("\n")


################################################
# 3. RANDOM PERMUTATION (equivalent to R sample(5))
################################################

perm = np.random.permutation(5) + 1
print("Random permutation of 1–5:")
print(perm)
print("\n")


################################################
# 4. OPTIMIZATION USING SCIPY minimize()
################################################

def mse(theta, y, x):
    """Mean squared error for linear model fit."""
    yhat = theta[0] + theta[1] * x
    return np.mean((y - yhat)**2)

# Example dataset for optimization
x_data = np.linspace(1, 10, 50)
y_data = 3 + 2 * x_data + np.random.normal(0, 1, size=len(x_data))

init = np.array([0, 1])

res = minimize(mse, init, args=(y_data, x_data))

print("Optimization results for linear model:")
print(res)
print("\n")


################################################
# 5. QUANTILE TRANSFORM METHOD (SIMULATION)
################################################

# Generate uniform(0,1)
u = np.random.uniform(size=1000)

# Convert uniform → normal using inverse CDF
normal_data = norm.ppf(u)

print("Generated 1000 Normal(0,1) values using inverse CDF:")
print(normal_data[:10])  # Show first 10
print("\n")


################################################
# END OF COMBINED PYTHON SCRIPT
################################################
