using Statistics
using Random
using HypothesisTests
using Plots

# --- Original Data (chickwts: soybean vs. linseed) ---
# Manual data definition based on R's built-in 'chickwts' dataset
X = [219, 271, 258, 248, 240, 246, 254, 301, 280, 236, 234, 309, 253, 303] # Soybean (n=14)
Y = [148, 221, 203, 224, 250, 253, 269, 272, 237, 244, 253, 259] # Linseed (n=12)
n1 = length(X) # 14
n2 = length(Y) # 12
Z = vcat(X, Y)
N = length(Z) # 26

# 1. Calculate the Observed T-statistic
# Julia's ttest() for two samples returns a field 'statistic'
# Welch's t-test (unequal variance) is appropriate here.
t0 = UnequalVarianceTTest(X, Y).statistic
println("Observed T-statistic (Welch's): $(round(t0, digits=4))")

# 2. Perform Permutation Resampling
B = 10000 
reps = zeros(B)
Random.seed!(42)
K = 1:N

for i in 1:B
    # Randomly choose n1 indices for the new 'X' group
    k = sample(K, n1, replace=false)
    
    x1 = Z[k]
    
    # The remaining indices form the new 'Y' group
    y1_mask = [!(i in k) for i in K]
    y1 = Z[y1_mask]
    
    # Calculate the T-statistic for the permuted data
    reps[i] = UnequalVarianceTTest(x1, y1).statistic
end

# 3. Calculate the P-value (One-sided: P(Stat_random >= Stat_observed))
pvalue = mean(vcat([t0], reps) .>= t0)

println("Permutation P-value (B=$B): $(round(pvalue, digits=6))")

# 4. Plotting Results
p = histogram(reps, bins=30, normalize = :pdf, label="Permutation Distribution", 
              title="Permutation Distribution for T-statistic (chickwts)", 
              xlabel="T-statistic", ylabel="Density", legend=:topleft)

vline!(p, [t0], line=(:blue, 2, :dash), label="Observed T-stat ($(round(t0, digits=2)))")
display(p)