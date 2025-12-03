# lidar_analysis.R
library(dplyr)
library(ggplot2)
library(caret)

set.seed(42)

# -----------------------------
# 1. Load and split data
# -----------------------------
lidar <- read.csv("lidar.csv")

# Train-test split (70/30)
train_idx <- createDataPartition(lidar$logratio, p = 0.7, list = FALSE)
lidar_train <- lidar[train_idx, ]
lidar_test  <- lidar[-train_idx, ]

# -----------------------------
# 2. Define models (degrees 1, 3, 5)
# -----------------------------
degrees <- c(1, 3, 5)

for (i in seq_along(degrees)) {
  d <- degrees[i]
  
  # Polynomial regression
  formula <- as.formula(paste("logratio ~ poly(range,", d, ", raw = TRUE)"))
  model <- lm(formula, data = lidar_train)
  
  # Predictions + errors
  y_hat <- predict(model, newdata = lidar_test)
  
  sse <- sum((lidar_test$logratio - y_hat)^2)
  mse <- mean((lidar_test$logratio - y_hat)^2)
  
  cat("Model obj", i, "(degree=", d, "): SSE=", round(sse, 4), 
      ", MSE=", round(mse, 4), "\n", sep = "")
}

# -----------------------------
# 3. Cross-validation smoother (LOESS)
# -----------------------------
spans <- seq(0.1, 1.0, length.out = 10)
cv_errors <- numeric(length(spans))

for (i in seq_along(spans)) {
  s <- spans[i]
  
  model <- loess(logratio ~ range, data = lidar_train, span = s)
  y_pred <- predict(model, newdata = lidar_test$range)
  
  cv_errors[i] <- mean((lidar_test$logratio - y_pred)^2)
}

# Best span
s_best <- spans[which.min(cv_errors)]
cat("\nBest smoother span (cross-validation):", s_best, "\n")

# -----------------------------
# 4. Fit with best span and plot
# -----------------------------
model_best <- loess(logratio ~ range, data = lidar, span = s_best)
lidar$y_smooth <- predict(model_best, newdata = lidar$range)

# Plot
p <- ggplot(lidar, aes(x = range, y = logratio)) +
  geom_point(alpha = 0.6) +
  geom_line(aes(y = y_smooth), color = "red", size = 1) +
  labs(
    title = paste0("Lidar Data — LOESS Smoothing (span=", round(s_best, 2), ")"),
    x = "range",
    y = "logratio"
  )

ggsave("r_lidar_smoother.png", plot = p, width = 7, height = 5)

cat("Saved r_lidar_smoother.png\n")
