# density_estimation.R

library(ggplot2)
library(stats)
library(KernSmooth)

set.seed(42)

# ---------------------------
# 1️⃣ Generate synthetic data
# ---------------------------
data <- c(rnorm(200, mean = 2, sd = 0.3),
          rnorm(200, mean = 4.5, sd = 0.5))


# ---------------------------
# 2️⃣ Histogram
# ---------------------------
ggplot(data.frame(x = data), aes(x)) +
  geom_histogram(aes(y = ..density..), bins = 20, fill = "lightblue", color = "black") +
  labs(title = "Histogram of Eruption Lengths",
       x = "Eruption Length", y = "Density")


# ---------------------------
# 3️⃣ Empirical CDF
# ---------------------------
sorted_data <- sort(data)
n <- length(data)
ecdf_vals <- ecdf(data)

ggplot(data.frame(x = sorted_data, y = ecdf_vals(sorted_data)),
       aes(x, y)) +
  geom_line(color = "blue", size = 1.2) +
  labs(title = "Empirical CDF",
       x = "Eruption Length", y = "F(x)")


# ---------------------------
# 4️⃣ Naive (Box) Density Estimator
# ---------------------------
naive_density_estimator <- function(x, data, h) {
  n <- length(data)
  fhat <- numeric(length(x))
  for (i in seq_along(x)) {
    fhat[i] <- sum(abs(data - x[i]) < h) / (2 * h)
  }
  return(fhat / n)
}

x_grid <- seq(min(data), max(data), length.out = 100)
fhat_naive <- naive_density_estimator(x_grid, data, h = 0.3)

ggplot(data.frame(x = x_grid, f = fhat_naive),
       aes(x, f)) +
  geom_line(color = "red", size = 1.2) +
  labs(title = "Naive Density Estimator",
       x = "Eruption Length", y = "Density")


# ---------------------------
# 5️⃣ Kernel Density Estimation (Gaussian)
# ---------------------------
kde_est <- bkde(data, bandwidth = 0.3)

ggplot(data.frame(x = kde_est$x, f = kde_est$y),
       aes(x, f)) +
  geom_line(color = "green", size = 1.2) +
  labs(title = "Kernel Density Estimate (h = 0.3)",
       x = "Eruption Length", y = "Density")


# ---------------------------
# 6️⃣ Cross-validation for bandwidth selection
# ---------------------------
bandwidths <- seq(0.1, 1.0, by = 0.1)

cv_error <- function(data, h) {
  n <- length(data)
  errors <- numeric(n)
  
  for (i in 1:n) {
    # Leave-one-out sample
    loo <- data[-i]
    kde <- bkde(loo, bandwidth = h)
    
    # Density at omitted point (nearest x-grid)
    idx <- which.min(abs(kde$x - data[i]))
    dens <- kde$y[idx]
    
    errors[i] <- (1 - dens)^2
  }
  
  mean(errors)
}

errors <- sapply(bandwidths, function(h) cv_error(data, h))

best_h <- bandwidths[which.min(errors)]
cat("Optimal bandwidth (cross-validation):", round(best_h, 3), "\n")

# Plot CV errors
ggplot(data.frame(h = bandwidths, err = errors),
       aes(h, err)) +
  geom_line(color = "orange", linewidth = 1.2) +
  labs(title = "Cross-Validation for Bandwidth Selection",
       x = "Bandwidth (h)", y = "CV Error")


# ---------------------------
# Final KDE with optimal bandwidth
# ---------------------------
final_kde <- bkde(data, bandwidth = best_h)

ggplot(data.frame(x = final_kde$x, f = final_kde$y),
       aes(x, f)) +
  geom_line(color = "purple", linewidth = 1.2) +
  labs(title = paste0("KDE with Optimal Bandwidth (h=", round(best_h, 2), ")"),
       x = "Eruption Length", y = "Density")
