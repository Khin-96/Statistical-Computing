import numpy as np
import scipy.stats as st

# --- AR(1) Simulation ---
def ar1_step(current_x, phi, sigma):
    """Generates the next step of an AR(1) process: X_t = phi * X_{t-1} + epsilon_t"""
    epsilon = np.random.normal(0, sigma)
    return phi * current_x + epsilon

# --- Fixed-Width Stopping Rule ---
def fixed_width_stopping(phi, sigma, initial_x, target_half_width, confidence_level=0.95, max_iter=1_000_000):
    
    # 1. Theoretical Stationary Variance and Autocorrelation Time
    # Stationary variance: Var(X_t) = sigma^2 / (1 - phi^2)
    stationary_variance = sigma**2 / (1 - phi**2)
    # Autocorrelation time (tau) for AR(1): (1 + phi) / (1 - phi)
    tau = (1 + phi) / (1 - phi)
    
    # Z-score for the confidence level (e.g., 1.96 for 95%)
    z_score = st.norm.ppf(1 - (1 - confidence_level) / 2)
    
    # 2. Initialize
    samples = [initial_x]
    current_x = initial_x
    
    print(f"AR(1) parameters: phi={phi}, sigma={sigma}. Theoretical tau={tau:.2f}")
    print(f"Target half-width: {target_half_width}. Confidence: {confidence_level*100:.0f}%")
    
    for n in range(1, max_iter + 1):
        # Generate the next sample
        current_x = ar1_step(current_x, phi, sigma)
        samples.append(current_x)
        
        # Check stopping condition every M steps (e.g., 100)
        if n % 100 == 0:
            N = len(samples)
            # Estimate of Var(Mean) = Var(X_t) * tau / N
            # Since Var(X_t) and tau are known for AR(1), we can use the formula directly.
            # Using the known (or estimated) tau for the variance of the sample mean:
            # Var(Mean) = stationary_variance * tau / N
            # S.E. of Mean = sqrt(Var(Mean))
            std_error_mean = np.sqrt(stationary_variance * tau / N)
            
            # Half-width of the confidence interval
            half_width = z_score * std_error_mean
            
            # Print progress (optional)
            if n % 10000 == 0:
                print(f"Iter: {n}, Half-width: {half_width:.6f}")
                
            # Stopping check
            if half_width <= target_half_width:
                print("-" * 30)
                print(f"*** STOPPING CONDITION MET ***")
                print(f"Total Samples (N): {N}")
                print(f"Final Half-width: {half_width:.6f}")
                print(f"Sample Mean: {np.mean(samples):.6f}")
                print(f"Effective Sample Size (ESS) estimate: {N / tau:.0f}")
                return np.array(samples)

    print(f"*** MAX ITERATIONS ({max_iter}) REACHED. STOPPING. ***")
    return np.array(samples)

# --- Run Simulation ---
PHI = 0.95      # AR(1) coefficient (must be < 1 for stationarity)
SIGMA = 1.0     # Noise standard deviation
INITIAL_X = 0.0 # Starting value (stationary mean is 0)
DELTA = 0.01    # Target half-width for the confidence interval

# Ensure PHI is in (-1, 1) for a stationary process
if abs(PHI) >= 1.0:
    raise ValueError("AR(1) parameter |phi| must be < 1 for stationarity.")

np.random.seed(42)
ar1_chain = fixed_width_stopping(PHI, SIGMA, INITIAL_X, DELTA)