using Random
using Statistics
using Distributions
using Plots
using HypothesisTests # For the Kolmogorov-Smirnov test

function bmnormal(n::Int, mu::Float64=0.0, sd::Float64=1.0)
    """
    Simulates n draws from a Normal(mu, sd) distribution using the 
    Box-Muller transformation.

    Parameters:
    n (Int): Number of draws to simulate.
    mu (Float64): Mean of the target Normal distribution.
    sd (Float64): Standard deviation of the target Normal distribution.

    Returns:
    Vector{Float64}: Array of n normally distributed random variables.
    """
    if n <= 0 || sd <= 0.0
        throw(DomainError("n and sd must be positive."))
    end
    
    # We need ceil(n / 2) pairs of uniform random numbers
    m = Int(ceil(n / 2))
    
    # 1. Generate two sets of independent uniform variables U1 and U2 in [0, 1)
    U1 = rand(m)
    U2 = rand(m)
    
    # 2. Apply the Box-Muller transformation
    R = sqrt.(-2.0 * log.(U1))
    Theta = 2.0 * π * U2
    
    # Z1 and Z2 are independent standard normal deviates (N(0, 1))
    Z1 = R .* cos.(Theta)
    Z2 = R .* sin.(Theta)
    
    # 3. Combine and truncate the results to n draws
    Z_combined = vcat(Z1, Z2)
    Z = Z_combined[1:n]
    
    # 4. Scale and shift to the target Normal(mu, sd) distribution
    X = mu .+ sd .* Z
    
    return X
end

# --- Simulation and Verification ---
const N_DRAWS = 2000
const MU = 10.0
const SD = 3.0

# Generate samples
Random.seed!(42)
sim_samples = bmnormal(N_DRAWS, MU, SD)

println("--- Simulation Results (N=$N_DRAWS, μ=$MU, σ=$SD) ---")
println("Sample Mean: $(round(mean(sim_samples), digits=4))")
println("Sample Std Dev: $(round(std(sim_samples), digits=4))")

## Plotting to be "convinced" 🖼️
# 1. Histogram of Samples vs. True PDF
p = histogram(sim_samples, bins=30, normalize = :pdf, label="Box-Muller Samples",
              title="Box-Muller Generated Samples vs. True Normal Distribution",
              xlabel="Value (X)", ylabel="Density", legend=:topleft)

# True Normal Distribution PDF
target_dist = Normal(MU, SD)
x_range = range(MU - 4*SD, stop=MU + 4*SD, length=100)
plot!(p, x_range, x -> pdf(target_dist, x), linewidth=2, linecolor=:red, label="True Normal PDF")
display(p)

## Statistical Test 🧪
# 2. Kolmogorov-Smirnov (K-S) Test for Normality
# We use the OneSampleADTest (Anderson-Darling) which is often preferred for normality testing in Julia
ad_test = OneSampleADTest(sim_samples, target_dist)

println("\n--- Statistical Test (Anderson-Darling) ---")
println("A-D Statistic: $(round(ad_test.A, digits=4))")
println("P-value: $(round(pvalue(ad_test), digits=4))")

if pvalue(ad_test) > 0.05
    println("Conclusion: Since p > 0.05, we do not reject H0. The samples appear to be Normally distributed.")
else
    println("Conclusion: Since p <= 0.05, we reject H0. The samples are unlikely to be Normally distributed.")
end