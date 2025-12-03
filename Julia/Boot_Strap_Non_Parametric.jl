using Statistics
using Random
using Distributions

# --- Sample Data (mtcars data for simplicity) ---
const data = [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 
              17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 
              21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 
              15.0, 21.4]
const N = length(data)

# --- Statistic Function ---
# Calculate the median of the data (or any other statistic)
calculate_statistic(sample) = median(sample)

# 1. Calculate the Observed Statistic
obs_stat = calculate_statistic(data)

# 2. Perform Bootstrap Resampling
B = 1000 # Number of bootstrap replicates
bootstrap_stats = zeros(B)

Random.seed!(42)

for i in 1:B
    # Resample with replacement from the original data
    # sample(data, N, replace=true) draws N elements
    bootstrap_sample = rand(data, N)
    bootstrap_stats[i] = calculate_statistic(bootstrap_sample)
end

# 3. Calculate Confidence Interval (Percentile Method)
confidence_level = 0.95
lower_quantile = (1 - confidence_level) / 2 
upper_quantile = (1 + confidence_level) / 2

# Julia's quantile function
ci_percentile = quantile(bootstrap_stats, [lower_quantile, upper_quantile])

# 4. Calculate Bias and Standard Error
bias = mean(bootstrap_stats) - obs_stat
std_error = std(bootstrap_stats)

println("Observed Statistic (Median): $(round(obs_stat, digits=4))")
println("--- Bootstrap Statistics ---")
println("Bias: $(round(bias, digits=4))")
println("Standard Error: $(round(std_error, digits=4))")
println("$(round(confidence_level*100))% Percentile CI: $(round.(ci_percentile, digits=4))")