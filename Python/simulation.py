################################################
# COMBINED PYTHON SCRIPT  
# Monte Carlo • Bootstrap • Simulation • Optimization  
# Numpy + SciPy versions of all R code  
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
np.column_stack([bootstrap_resample(np.arange(6, 11)) for _ in range(5)])


################################################
# 2. BOOTSTRAP TWO-SAMPLE DIFFERENCE IN MEANS
################################################

def diff_in_means(x, y):
    """Difference in sample means."""
    return np.mean(x) - np.mean(y)

# Example data (replace with real x, y)
# x = np.random.normal(size=20)
# y = np.random.normal(size=25)

B = 1000
boot_stats = []

# Bootstrap loop
for _ in range(B):
    x_star = np.random.choice(x, size=len(x), replace=True)
    y_star = np.random.choice(y, size=len(y), replace=True)
    boot_stats.append(diff_in_means(x_star, y_star))

# 95% CI
np.quantile(boot_stats, [0.025, 0.975])


################################################
# 3. RANDOM PERMUTATION (equivalent to R sample(5))
################################################

np.random.permutation(5) + 1  # add 1 because Python starts at 0


################################################
# 4. OPTIMIZATION USING SCIPY minimize()
################################################

# Example: load gmp.dat (must exist locally)
# gmp = np.genfromtxt("gmp.dat")

# Example: create variable pop = gmp / pcgmp
# pop = gmp[:, 0] / gmp[:, 1]

def mse(theta, y, x):
    """Mean squared error for linear model fit."""
    yhat = theta[0] + theta[1] * x
    return np.mean((y - yhat)**2)

# Initial guess
init = np.array([0, 1])

# Example optimization call:
# res = minimize(mse, init, args=(gmp[:,1], pop))


################################################
# 5. QUANTILE TRANSFORM METHOD (SIMULATION)
################################################

# Generate uniform(0,1)
u = np.random.uniform(size=1000)

# Transform using inverse Normal CDF
x = norm.ppf(u)


################################################
# END OF COMBINED PYTHON SCRIPT
################################################
