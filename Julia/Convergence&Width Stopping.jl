using Distributions
using Random
using Statistics
using SpecialFunctions: erfinv

# --- AR(1) Simulation ---
function ar1_step(current_x, phi, sigma)
    """Generates the next step of an AR(1) process: X_t = phi * X_{t-1} + epsilon_t"""
    epsilon = rand(Normal(0, sigma))
    return phi * current_x + epsilon
end

# --- Fixed-Width Stopping Rule ---
function fixed_width_stopping(phi, sigma, initial_x, target_half_width, confidence_level=0.95, max_iter=1_000_000)
    
    # 1. Theoretical Stationary Variance and Autocorrelation Time
    # Stationary variance: Var(X_t) = sigma^2 / (1 - phi^2)
    stationary_variance = sigma^2 / (1 - phi^2)
    # Autocorrelation time (tau) for AR(1): (1 + phi) / (1 - phi)
    tau = (1 + phi) / (1 - phi)
    
    # Z-score for the confidence level (e.g., 1.96 for 95%)
    # Z-score = quantile(Normal(0,1), 1 - (1 - confidence_level) / 2)
    # Using erfinv: z = sqrt(2) * erfinv(2 * p - 1)
    z_score = sqrt(2.0) * erfinv(2 * (1 - (1 - confidence_level) / 2) - 1)
    
    # 2. Initialize
    samples = Float64[]
    current_x = initial_x
    
    println("AR(1) parameters: phi=$phi, sigma=$sigma. Theoretical tau=$(round(tau, digits=2))")
    println("Target half-width: $target_half_width. Confidence: $(round(confidence_level*100))%")
    
    for n in 1:max_iter
        # Generate the next sample
        current_x = ar1_step(current_x, phi, sigma)
        push!(samples, current_x)
        
        # Check stopping condition every M steps (e.g., 100)
        if n % 100 == 0
            N = length(samples)
            # Estimate of Var(Mean) = Var(X_t) * tau / N
            # S.E. of Mean = sqrt(Var(Mean))
            std_error_mean = sqrt(stationary_variance * tau / N)
            
            # Half-width of the confidence interval
            half_width = z_score * std_error_mean
            
            # Print progress (optional)
            if n % 10000 == 0
                println("Iter: $n, Half-width: $(round(half_width, digits=6))")
            end
                
            # Stopping check
            if half_width <= target_half_width
                println("-" * 30)
                println("*** STOPPING CONDITION MET ***")
                println("Total Samples (N): $N")
                println("Final Half-width: $(round(half_width, digits=6))")
                println("Sample Mean: $(round(mean(samples), digits=6))")
                println("Effective Sample Size (ESS) estimate: $(round(N / tau, digits=0))")
                return samples
            end
        end
    end

    println("*** MAX ITERATIONS ($max_iter) REACHED. STOPPING. ***")
    return samples
end

# --- Run Simulation ---
PHI = 0.95      # AR(1) coefficient (must be < 1 for stationarity)
SIGMA = 1.0     # Noise standard deviation
INITIAL_X = 0.0 # Starting value (stationary mean is 0)
DELTA = 0.01    # Target half-width for the confidence interval

# Ensure PHI is in (-1, 1) for a stationary process
if abs(PHI) >= 1.0
    error("AR(1) parameter |phi| must be < 1 for stationarity.")
end

Random.seed!(42)
ar1_chain = fixed_width_stopping(PHI, SIGMA, INITIAL_X, DELTA)