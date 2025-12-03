using Statistics
using Random
using HypothesisTests # Not strictly needed for correlation, but good practice for stats
using Plots

# --- Original Data (SAT-type Score vs. SAT Score) ---
Score = [58, 48, 48, 41, 34, 43, 38, 53, 41, 60, 55, 44,  
         43, 49, 47, 33, 47, 40, 46, 53, 40, 45, 39, 47,  
         50, 53, 46, 53]
SAT = [590, 590, 580, 490, 550, 580, 550, 700, 560, 690, 800, 600, 
       650, 580, 660, 590, 600, 540, 610, 580, 620, 600, 560, 560, 
       570, 630, 510, 620]
N = length(Score)

# 1. Calculate the Observed Correlation Coefficient (r)
r_obt = cor(Score, SAT)
println("Observed Correlation (r): $(round(r_obt, digits=6))")

# 2. Perform Permutation Resampling
B = 10000 
r_random = zeros(B)
Random.seed!(42)

for i in 1:B
    # Randomly shuffle (permute) the SAT scores
    X_permuted = shuffle(SAT)
    Y = Score # Keep Score fixed
    
    # Calculate the correlation for the permuted data
    r_random[i] = cor(Y, X_permuted)
end

# 3. Calculate the P-value (One-sided: P(Stat_random >= Stat_observed))
pvalue = mean(vcat([r_obt], r_random) .>= r_obt)

println("Permutation P-value (B=$B): $(round(pvalue, digits=6))")

# 4. Plotting Results
p = histogram(r_random, bins=30, normalize = :pdf, label="Permutation Distribution", 
              title="Permutation Distribution for Correlation Coefficient", 
              xlabel="Correlation Coefficient (r)", ylabel="Density", legend=:topleft)

vline!(p, [r_obt], line=(:navy, 2, :dash), label="Observed r ($(round(r_obt, digits=4)))")
display(p)