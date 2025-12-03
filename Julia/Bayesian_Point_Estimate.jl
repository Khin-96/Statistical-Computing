using Distributions
using Printf

# --- Input Parameters (Prior and Data) ---
alpha1 = 0.5  # Prior Alpha parameter
alpha2 = 0.5  # Prior Beta parameter
x = 2         # Observed successes (data)
n = 10        # Total trials (data)

# --- Posterior Distribution Parameters ---
# Posterior is Beta(alpha1', alpha2')
posterior_a = x + alpha1        # alpha1' = 2.5
posterior_b = n - x + alpha2    # alpha2' = 8.5

posterior_dist = Beta(posterior_a, posterior_b)

@printf("Prior: Beta(%.1f, %.1f) | Data: x=%d, n=%d\n", alpha1, alpha2, x, n)
@printf("Posterior: Beta(%.1f, %.1f)\n\n", posterior_a, posterior_b)

# --- 1. Posterior Mean ---
# mean(dist) calculates E[p|x]
posterior_mean = mean(posterior_dist)

# --- 2. Posterior Median ---
# quantile(dist, 0.5) is the 50th percentile (qbeta in R, ppf in Python)
posterior_median = quantile(posterior_dist, 0.5)

@printf("--- Bayesian Point Estimates ---\n")
@printf("Posterior Mean (E[p|x]): %.7f\n", posterior_mean)
@printf("Posterior Median:        %.7f\n", posterior_median)
# Expected output: 0.2272727 and 0.2103736