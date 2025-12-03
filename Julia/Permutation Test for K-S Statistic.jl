using Statistics
using Random
using HypothesisTests # Required for the Kolmogorov-Smirnov test
using Plots

# --- Data Re-use (chickwts: soybean vs. linseed) ---
X = [219, 271, 258, 248, 240, 246, 254, 301, 280, 236, 234, 309, 253, 303] # Soybean (n=14)
Y = [148, 221, 203, 224, 250, 253, 269, 272, 237, 244, 253, 259] # Linseed (n=12)
n1 = length(X) 
n2 = length(Y) 
Z = vcat(X, Y)
N = length(Z) # 26

# 1. Calculate the Observed K-S Statistic (D)
# Julia's KS_Test returns a field 'D' for the statistic
D_obs = TwoSampleKSTest(X, Y).D
println("Observed K-S Statistic (D): $(round(D_obs, digits=4))")

# 2. Perform Permutation Resampling
B = 10000 
D_stats = zeros(B)
Random.seed!(42)
K = 1:N

for i in 1:B
    # Randomly choose n1 indices for the new 'X' group
    k = sample(K, n1, replace=false)
    
    x1 = Z[k]
    
    # The remaining indices form the new 'Y' group
    y1_mask = [!(i in k) for i in K]
    y1 = Z[y1_mask]
    
    # Calculate the K-S statistic for the permuted data
    D_stats[i] = TwoSampleKSTest(x1, y1).D
end

# 3. Calculate the P-value (One-sided: P(Stat_random >= Stat_observed))
pvalue = mean(vcat([D_obs], D_stats) .>= D_obs)

println("Permutation P-value (B=$B): $(round(pvalue, digits=6))")

# 4. Plotting Results
p = histogram(D_stats, bins=30, normalize = :pdf, label="Permutation Distribution", 
              title="Permutation Distribution for K-S Statistic", 
              xlabel="K-S Statistic (D)", ylabel="Density", legend=:topleft)

vline!(p, [D_obs], line=(:darkviolet, 2, :dash), label="Observed D Stat ($(round(D_obs, digits=2)))")
display(p)