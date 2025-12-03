###############################
# Bayesian Statistics in R
# Posterior Mean, Median,
# Credible Intervals,
# Hypothesis Testing, Bayes Factor
###############################

# Parameters
alpha1 <- alpha2 <- 1/2
x <- 2
n <- 10

###############################
# 1. Posterior Mean
###############################
posterior_mean <- (x + alpha1) / (n + alpha1 + alpha2)
posterior_mean

###############################
# 2. Posterior Median
###############################
posterior_median <- qbeta(0.5, x + alpha1, n - x + alpha1)
posterior_median

###############################
# 3. 80% Credible Interval
###############################
credible_interval <- cbind(
  qbeta(0.1, x + alpha1, n - x + alpha2),
  qbeta(0.9, x + alpha1, n - x + alpha2)
)
credible_interval

###############################
# 4. Bayesian Hypothesis Test
# H: p ≤ 0.5  vs  p > 0.5
###############################
p_less <- pbeta(0.5, x + alpha1, n - x + alpha2)
p_greater <- pbeta(0.5, x + alpha1, n - x + alpha2, lower.tail = FALSE)

p_less
p_greater

###############################
# 5. Bayes Factor
###############################
p0 <- 1/2

# Likelihood under H0
b1 <- dbinom(x, n, p0)

# Marginal likelihood under prior
b2 <- choose(n, x) *
  beta(x + alpha1, n - x + alpha2) /
  beta(alpha1, alpha2)

BayesFactor <- b1 / b2
BayesFactor

###############################
# 6. Frequentist p-value
###############################
pvalue <- 2 * pbinom(x, n, p0)
pvalue
