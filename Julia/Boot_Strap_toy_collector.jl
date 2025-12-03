using Statistics
using Random
using Distributions
using Plots

# --- Problem Setup ---
# Probabilities for each of the 15 unique toys (sum to 1.0)
const prob_table = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]
const boxes = 1:15 # Toy IDs from 1 to 15
const N_toys = length(prob_table)

function box_count(prob=prob_table, toy_ids=boxes)
    """
    Simulates the process of collecting all N_toys.
    Returns the total number of boxes opened (i).
    """
    # Use a BitSet or a Boolean array to track collected toys
    # A Boolean array initialized to false works well here
    collected = falses(N_toys) 
    i = 0 # Box count
    
    while sum(collected) < N_toys
        # 1. Sample one toy based on the probability distribution
        # Julia's StatsBase.sample (or similar) is generally used for this, 
        # but a manual approach using discrete sampling is reliable.
        # Since we just need to use `rand(::Categorical)`, we need the Distributions package.
        
        # The index (1 to 15) corresponds to the toy ID
        toy_index = rand(Categorical(prob))
        
        # 2. Mark the toy as collected
        collected[toy_index] = true
        
        # 3. Increment box count
        i += 1
    end
    
    return i
end

# --- Run Monte Carlo Simulation ---
TRIALS = 10000 
Random.seed!(42)
sim_boxes = zeros(TRIALS)

for i in 1:TRIALS
    sim_boxes[i] = box_count()
end

# --- Calculate Results ---
est = mean(sim_boxes)
mcse = std(sim_boxes) / sqrt(TRIALS) # Monte Carlo Standard Error
confidence_level = 0.95
# Z-score for 95% CI is approximately 1.96
z_score = quantile(Normal(), 1 - (1 - confidence_level) / 2) 
interval = est .+ [-1, 1] .* z_score .* mcse

println("Number of Trials (B): $TRIALS")
println("Estimated Mean Boxes (E[T]): $(round(est, digits=4))")
println("MC Standard Error (MCSE): $(round(mcse, digits=4))")
println("$(round(confidence_level*100))% Confidence Interval: $(round.(interval, digits=4))")

# --- Plotting Results ---
p = histogram(sim_boxes, bins=:auto, normalize=:pdf, label="MC Simulation",
              title="Histogram of Total Boxes Needed (Unequal Probabilities)",
              xlabel="Total Boxes Opened", ylabel="Density", legend=:topright)

vline!(p, [est], line=(:red, 2, :dash), label="Estimated Mean ($(round(est, digits=2)))")
display(p)