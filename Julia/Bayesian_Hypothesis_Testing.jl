using Distributions
using Printf

# --- Input Parameters (Prior and Data) ---
alpha1 = 0.5  # Prior Alpha parameter
alpha2 = 0.5  # Prior Beta parameter
x = 2         # Observed successes (data)
n = 10        # Total trials (data)

# --- Posterior Distribution Parameters ---
posterior_a = x + alpha1
posterior_b = n - x + alpha2

posterior_dist = Beta(posterior_a, posterior_b)

# --- Hypothesis Definitions ---
# H1: p < 0.5
# H0: p >= 0.5
p_threshold = 0.5

@printf("Prior: Beta(%.1f, %.1f) | Data: x=%d, n=%d\n", alpha1, alpha2, x, n)
@printf("Posterior: Beta(%.1f, %.1f)\n\n", posterior_a, posterior_b)

# --- Calculate Posterior Probabilities ---
# P(p < p_threshold | x) is found using the Cumulative Distribution Function (cdf)
prob_p_less_than_threshold = cdf(posterior_dist, p_threshold)

# P(p >= p_threshold | x)
prob_p_greater_equal_threshold = 1.0 - prob_p_less_than_threshold

@printf("--- Posterior Hypothesis Probabilities (Threshold p=%.1f) ---\n", p_threshold)
@printf("P(p < %.1f | x) (H1):    %.7f\n", p_threshold, prob_p_less_than_threshold)
@printf("P(p >= %.1f | x) (H0):   %.7f\n", p_threshold, prob_p_greater_equal_threshold)
# Expected output H1: 0.9739634, H0: 0.0260366