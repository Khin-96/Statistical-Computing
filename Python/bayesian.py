################################
# Bayesian Statistics in Python
# Posterior Mean, Median,
# Credible Intervals,
# Hypothesis Testing, Bayes Factor
################################

from scipy.stats import beta, binom
from math import comb

# Parameters
alpha1 = alpha2 = 0.5
x = 2
n = 10

################################
# 1. Posterior Mean
################################
posterior_mean = (x + alpha1) / (n + alpha1 + alpha2)
print("Posterior Mean:", posterior_mean)

################################
# 2. Posterior Median
################################
posterior_median = beta.ppf(0.5, x + alpha1, n - x + alpha1)
print("Posterior Median:", posterior_median)

################################
# 3. 80% Credible Interval
################################
lower = beta.ppf(0.1, x + alpha1, n - x + alpha2)
upper = beta.ppf(0.9, x + alpha1, n - x + alpha2)
print("80% Credible Interval:", (lower, upper))

################################
# 4. Bayesian Hypothesis Test
# H: p ≤ 0.5  vs  p > 0.5
################################
p_less = beta.cdf(0.5, x + alpha1, n - x + alpha2)
p_greater = 1 - p_less

print("P(p ≤ 0.5):", p_less)
print("P(p > 0.5):", p_greater)

################################
# 5. Bayes Factor
################################
p0 = 0.5

# Likelihood under H0
b1 = binom.pmf(x, n, p0)

# Marginal likelihood under prior
# (Using the Beta-binomial model)
b2 = comb(n, x) * (
    beta.pdf(0, x + alpha1, n - x + alpha2) /
    beta.pdf(0, alpha1, alpha2)
)

BayesFactor = b1 / b2
print("Bayes Factor:", BayesFactor)

################################
# 6. Frequentist p-value
################################
pvalue = 2 * binom.cdf(x, n, p0)
print("Frequentist p-value:", pvalue)
