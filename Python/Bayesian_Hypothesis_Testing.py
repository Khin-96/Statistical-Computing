import numpy as np
from scipy.stats import beta

# --- Input Parameters (Prior and Data) ---
alpha1 = 0.5  # Prior Alpha parameter
alpha2 = 0.5  # Prior Beta parameter
x = 2         # Observed successes (data)
n = 10        # Total trials (data)

# --- Posterior Distribution Parameters ---
posterior_a = x + alpha1
posterior_b = n - x + alpha2

# --- Hypothesis Definitions ---
# H1: p < 0.5
# H0: p >= 0.5
p_threshold = 0.5

# --- Calculate Posterior Probabilities ---
# P(p < p_threshold | x) is found using the Cumulative Distribution Function (cdf)
# In R this is pbeta(0.5, a', b')
prob_p_less_than_threshold = beta.cdf(p_threshold, posterior_a, posterior_b)

# P(p >= p_threshold | x)
prob_p_greater_equal_threshold = 1.0 - prob_p_less_than_threshold

print(f"Prior: Beta({alpha1}, {alpha2}) | Data: x={x}, n={n}")
print(f"Posterior: Beta({posterior_a}, {posterior_b})\n")
print(f"--- Posterior Hypothesis Probabilities (Threshold p={p_threshold}) ---")
print(f"P(p < {p_threshold} | x) (H1):    {prob_p_less_than_threshold:.7f}")
print(f"P(p >= {p_threshold} | x) (H0):   {prob_p_greater_equal_threshold:.7f}")
# Expected output H1: 0.9739634, H0: 0.0260366