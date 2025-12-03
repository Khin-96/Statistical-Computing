using Distributions
using Random
using Statistics
using Plots

# --- Target Distribution (Exp(1)) ---
# The target distribution is Exp(1). The unnormalized PDF is just exp(-x) for x >= 0.
const TARGET_DIST = Exponential(1)
target_logpdf(x) = logpdf(TARGET_DIST, x)

# --- Proposal Distribution (Symmetric Random Walk) ---
# q(x'|x) = N(x, step_size^2)
proposal_draw(current_x, step_size) = rand(Normal(current_x, step_size))

# --- Metropolis-Hastings Sampler ---
function metropolis_hastings(target_logpdf, proposal_draw, step_size, initial_x, n_samples)
    samples = zeros(n_samples)
    current_x = initial_x
    
    # Track acceptance rate
    accepted_count = 0
    
    for i in 1:n_samples
        # 1. Propose a candidate x'
        proposed_x = proposal_draw(current_x, step_size)
        
        # Ensure the proposed value is in the support of Exp(1) (x >= 0)
        # If not, the target_logpdf will return -Inf, and log_alpha will be -Inf
        if proposed_x < 0
            alpha = 0.0
        else
            # 2. Compute acceptance ratio $\alpha$ using log-densities
            # For a symmetric proposal, q ratio is 1, so log(q ratio) is 0.
            # log_alpha = min(0, log(pi(x')) - log(pi(x)))
            log_alpha = target_logpdf(proposed_x) - target_logpdf(current_x)
            alpha = exp(min(0.0, log_alpha))
        end

        # 3. Accept or Reject
        if rand() < alpha
            current_x = proposed_x # Accept
            accepted_count += 1
        end
            
        samples[i] = current_x # Store the current state (accepted or rejected)
    end
    
    return samples, accepted_count
end

# --- Run Simulation ---
N = 50000        # Number of samples
BURN_IN = 1000   # Number of samples to discard
STEP_SIZE = 1.0  # Proposal standard deviation (tuning parameter)
initial_x = 2.0  # Non-negative start point

Random.seed!(42) # For reproducibility
println("Running MH for Exp(1) with N=$N and step_size=$STEP_SIZE...")
mh_samples, accepted_count = metropolis_hastings(target_logpdf, proposal_draw, STEP_SIZE, initial_x, N)
final_samples = mh_samples[BURN_IN+1:end]

# --- Plotting Results (using Plots.jl) ---
p1 = histogram(final_samples, bins=50, normalize = :pdf, label="MH Samples (Normalized)", 
               title="Metropolis-Hastings Sampler for Exp(1)", xlabel="x", ylabel="Density", 
               xlim=(0, 8), legend=:topright)

# True Exp(1) PDF line
x_grid = 0:0.1:8
plot!(p1, x_grid, x -> pdf(TARGET_DIST, x), linewidth=2, linecolor=:red, label="True Exp(1) PDF")
display(p1)

# Print mean for comparison (True mean of Exp(1) is 1.0)
println("Sample Mean: $(mean(final_samples))")
println("Acceptance Rate: $(accepted_count / N)")