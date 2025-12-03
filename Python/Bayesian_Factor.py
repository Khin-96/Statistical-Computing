import numpy as np
from scipy.stats import binom
from scipy.special import beta as beta_func

# --- Input Parameters (Prior and Data) ---
alpha1 = 0.5  # Prior Alpha parameter
alpha2 = 0.5  # Prior Beta parameter
x = 2         # Observed successes (data)
n = 10        # Total trials (data)
p0 = 0.5      # Point null hypothesis: H0: p = p0

print(f"Prior: Beta({alpha1}, {alpha2}) | Data: x={x}, n={n} | H0: p={p0}\n")

# --- 1. Likelihood of x under H0 (P(x | p0)) ---
# This uses the Binomial PMF (dbinom in R)
likelihood_h0 = binom.pmf(k=x, n=n, p=p0)

# --- 2. Prior Predictive Likelihood P(x | Prior) ---
# P(x | Prior) = (n choose x) * [Beta(x + a1, n - x + a2) / Beta(a1, a2)]

# Calculate Binomial Coefficient (n choose x)
n_choose_x = binom.pmf(k=x, n=n, p=0.5) / (0.5**n) # Use p=0.5 for stability to calculate C(n,x)

# Calculate the Beta function parts: B(a,b) = Gamma(a)Gamma(b) / Gamma(a+b)
numerator_beta = beta_func(x + alpha1, n - x + alpha2)
denominator_beta = beta_func(alpha1, alpha2)

prior_predictive_likelihood = n_choose_x * (numerator_beta / denominator_beta)

# --- 3. Bayes Factor (BF01) ---
# BF01 = P(x | H0) / P(x | H1) = Likelihood_H0 / Prior_Predictive_Likelihood
bayes_factor = likelihood_h0 / prior_predictive_likelihood

# --- 4. Frequentist p-value (for comparison) ---
# Two-tailed p-value: 2 * P(X <= x)
p_value = 2 * binom.cdf(k=x, n=n, p=p0)

print("--- Bayes Factor (BF01) Calculation ---")
print(f"Likelihood P(x | H0: p={p0}): {likelihood_h0:.7f}")
print(f"Prior Predictive P(x | Prior): {prior_predictive_likelihood:.7f}")
print(f"Bayes Factor (BF01):         {bayes_factor:.7f}")
print(f"Frequentist p-value:         {p_value:.7f}")
# Expected output BF01: 0.5967366