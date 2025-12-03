using Distributions
using SpecialFunctions
using Printf

# --- Input Parameters (Prior and Data) ---
alpha1 = 0.5  # Prior Alpha parameter
alpha2 = 0.5  # Prior Beta parameter
x = 2         # Observed successes (data)
n = 10        # Total trials (data)
p0 = 0.5      # Point null hypothesis: H0: p = p0

@printf("Prior: Beta(%.1f, %.1f) | Data: x=%d, n=%d | H0: p=%.1f\n\n", alpha1, alpha2, x, n, p0)

# --- 1. Likelihood of x under H0 (P(x | p0)) ---
# This uses the Binomial PMF (pdf in Julia's Distributions)
likelihood_h0 = pdf(Binomial(n, p0), x)

# --- 2. Prior Predictive Likelihood P(x | Prior) ---
# P(x | Prior) = (n choose x) * [Beta(x + a1, n - x + a2) / Beta(a1, a2)]

# Calculate Binomial Coefficient (n choose x)
n_choose_x = binomial(n, x)

# Calculate the Beta function parts: B(a,b)
# This uses SpecialFunctions.beta_func
numerator_beta = beta_func(x + alpha1, n - x + alpha2)
denominator_beta = beta_func(alpha1, alpha2)

prior_predictive_likelihood = n_choose_x * (numerator_beta / denominator_beta)

# --- 3. Bayes Factor (BF01) ---
# BF01 = P(x | H0) / P(x | Prior)
bayes_factor = likelihood_h0 / prior_predictive_likelihood

# --- 4. Frequentist p-value (for comparison) ---
# Two-tailed p-value: 2 * P(X <= x)
# This uses the Cumulative Distribution Function (cdf in Julia's Distributions)
p_value = 2 * cdf(Binomial(n, p0), x)

@printf("--- Bayes Factor (BF01) Calculation ---\n")
@printf("Likelihood P(x | H0: p=%.1f): %.7f\n", p0, likelihood_h0)
@printf("Prior Predictive P(x | Prior): %.7f\n", prior_predictive_likelihood)
@printf("Bayes Factor (BF01):         %.7f\n", bayes_factor)
@printf("Frequentist p-value:         %.7f\n", p_value)
# Expected output BF01: 0.5967366