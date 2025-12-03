using Statistics
using Random
using Plots

# --- Original Data (DRP Scores) ---
const T = [24, 43, 58, 71, 61, 44, 67, 49, 59, 52, 62, 54, 46, 43, 57, 
           43, 57, 56, 53, 49, 33]
const C = [42, 43, 55, 26, 33, 41, 19, 54, 46, 10, 17, 60, 37, 42, 55, 
           28, 62, 53, 37, 42, 20, 48, 85]
const n1 = length(T)
const n2 = length(C)
const Z = vcat(T, C) # Pooled data
const N = length(Z)

# 1. Calculate the Observed Statistic
obs_stat = mean(T) - mean(C)
println("Observed Difference in Means: $(round(obs_stat, digits=4))")

# 2. Perform Permutation Resampling
B = 10000 
new_stats = zeros(B)
Random.seed!(42)

for i in 1:B
    # Choose n1 unique indices for the new 'Treatment' group
    idx = sample(1:N, n1, replace=false)
    
    newT = Z[idx]
    
    # Get the indices for the new 'Control' group
    control_mask = [!(i in idx) for i in 1:N]
    newC = Z[control_mask]
    
    new_stats[i] = mean(newT) - mean(newC)
end

# 3. Calculate the P-value (One-sided: P(Stat_random >= Stat_observed))
# Include the observed statistic for the (B+1) permutation P-value
pvalue = mean(vcat([obs_stat], new_stats) .>= obs_stat)

println("Permutation P-value (B=$B): $(round(pvalue, digits=6))")

# 4. Plotting Results
p = histogram(new_stats, bins=30, normalize = :pdf, label="Permutation Distribution", 
              title="Permutation Distribution for Difference in Means", 
              xlabel="Difference in Means (T - C)", ylabel="Density", legend=:topleft)

vline!(p, [obs_stat], line=(:red, 2, :dash), label="Observed Stat ($(round(obs_stat, digits=2)))")
display(p)