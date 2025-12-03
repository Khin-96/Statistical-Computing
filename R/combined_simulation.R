###############################################
# COMBINED R SCRIPT  
# Monte Carlo • Bootstrap • Simulation • Optimization  
# All code extracted & cleaned from your HTML files  
# Author: ChatGPT reconstruction  
###############################################

###############################################
# 1. BOOTSTRAP RESAMPLING FUNCTION
###############################################

# Bootstrap: Resample the same length with replacement
bootstrap.resample <- function(object) {
  sample(object, length(object), replace = TRUE)
}

# Example: Repeat bootstrap 5 times on values 6:10
replicate(5, bootstrap.resample(6:10))


###############################################
# 2. BOOTSTRAP TWO-SAMPLE DIFFERENCE IN MEANS
###############################################

# Function: difference in sample means
diff.in.means <- function(x, y) {
  mean(x) - mean(y)
}

# Example data (replace with your real x, y)
# x <- rnorm(20)
# y <- rnorm(25)

# Number of bootstrap replicates
B <- 1000

# Store bootstrap results
boot.stats <- numeric(B)

# Generate bootstrap distribution
for (i in 1:B) {
  x.star <- sample(x, replace = TRUE)
  y.star <- sample(y, replace = TRUE)
  boot.stats[i] <- diff.in.means(x.star, y.star)
}

# 95% bootstrap confidence interval
quantile(boot.stats, c(.025, .975))


###############################################
# 3. PERMUTATIONS USING sample()
###############################################

# Produce a random permutation of numbers 1 to 5
sample(5)


###############################################
# 4. OPTIMIZATION USING optim()
###############################################

# Load dataset (gmp.dat must be in working directory)
# gmp <- read.table("gmp.dat")

# Example derived variable
# gmp$pop <- gmp$gmp / gmp$pcgmp

# Mean squared error function to minimize
mse <- function(theta, y, x) {
  yhat <- theta[1] + theta[2] * x
  mean((y - yhat)^2)
}

# Initial parameter guess
init <- c(0, 1)

# Example optimization call:
# library(numDeriv)
# optim(init, mse, y = gmp$pcgmp, x = gmp$pop)


###############################################
# 5. QUANTILE TRANSFORM METHOD (SIMULATION)
###############################################

# Generate uniform(0,1)
u <- runif(1000)

# Transform to normal via inverse CDF
x <- qnorm(u)


###############################################
# END OF COMBINED R SCRIPT
###############################################
