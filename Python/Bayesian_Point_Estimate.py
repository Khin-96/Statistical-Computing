import numpy as np
from scipy.stats import beta

# --- Input Parameters (Prior and Data) ---
alpha1 = 0.5  # Prior Alpha parameter
alpha2 = 0.5  # Prior Beta parameter
x = 2         # Observed successes (data)
n = 10        # Total trials (data)

# --- Posterior Distribution Parameters ---
# Posterior is Beta(alpha1', alpha2')
posterior_a = x + alpha1        # alpha1' = 2 + 0.5 = 2.5
posterior_b = n - x + alpha2    # alpha2' = (10 - 2) + 0.5 = 8.5

print(f"Prior: Beta({alpha1}, {alpha2}) | Data: x={x}, n={n}")
print(f"Posterior: Beta({posterior_a}, {posterior_b})\n")

# --- 1. Posterior Mean ---
# The mean of a Beta(a, b) distribution is a / (a + b)
posterior_mean = posterior_a / (posterior_a + posterior_b)

# --- 2. Posterior Median ---
# The median is the 50th percentile, found using the Quantile Function (ppf)
# In R this is qbeta(0.5, a', b')
posterior_median = beta.ppf(0.5, posterior_a, posterior_b)

print("--- Bayesian Point Estimates ---")
print(f"Posterior Mean (E[p|x]): {posterior_mean:.7f}")
print(f"Posterior Median:        {posterior_median:.7f}")
# Expected output: 0.2272727 and 0.2103736