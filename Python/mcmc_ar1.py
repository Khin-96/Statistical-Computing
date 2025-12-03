import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

# --- 1. MCMC Helper Functions ---

def ar1_step(m: float, rho: float, tau: float) -> float:
    """
    Generates the next observation from the AR(1) chain.
    X_i = rho * X_{i-1} + epsilon_i, where epsilon_i ~ N(0, tau^2)
    """
    return rho * m + np.random.normal(0, tau)

def ar1_gen(mc: list, p: int, rho: float, tau: float) -> list:
    """
    Extends an existing Markov Chain (MC) by 'p' steps.
    """
    current_state = mc[-1] if mc else 0.0 # Start at 0 if chain is empty
    
    for _ in range(p):
        current_state = ar1_step(current_state, rho, tau)
        mc.append(current_state)
    return mc

def mcse_batch_means(chain: np.ndarray, batch_size: int = None) -> float:
    """
    Calculates the Monte Carlo Standard Error (MCSE) using the Batch Means method.
    The batch size is typically floor(sqrt(N)).
    Returns the standard error of the mean estimate.
    """
    N = len(chain)
    if N < 10:
        return np.inf # Cannot reliably calculate MCSE on tiny chain

    if batch_size is None:
        batch_size = max(1, int(np.floor(np.sqrt(N))))
    
    # K = number of batches
    K = N // batch_size
    
    if K < 2:
        # Not enough batches, use a smaller batch size or return estimate
        if batch_size > 1:
            return mcse_batch_means(chain, batch_size=batch_size // 2)
        return np.std(chain, ddof=1) / np.sqrt(N) # Fallback to naive SE

    # Truncate the chain to be evenly divisible into K batches
    truncated_chain = chain[:K * batch_size]
    
    # Reshape the chain to (K batches, batch_size)
    batches = truncated_chain.reshape(K, batch_size)
    
    # Calculate the mean of each batch
    batch_means = np.mean(batches, axis=1)
    
    # Calculate the variance of the batch means
    # The estimated variance of the overall mean is: 
    # (batch_size / K) * variance(batch_means)
    # The MCSE is the square root of the estimated variance of the mean.
    # Note: np.std(..., ddof=1) calculates standard deviation (divides by K-1)
    mcse = (np.std(batch_means, ddof=1) / np.sqrt(K)) * np.sqrt(batch_size)
    
    return mcse

# --- 2. Simulation Parameters ---
np.random.seed(20) # for reproducibility
tau = 1.0           # StdDev of the noise epsilon
rho = 0.95          # Autoregressive coefficient
eps = 0.1           # Target half-width for the confidence interval (w_n <= 0.2)
start = 1000        # Initial burn-in/run length
r = 1000            # Iterations to add in subsequent runs
confidence_level = 0.95
alpha = 1 - confidence_level

# --- 3. Stopping Rule Execution ---
out = []
# Initialize with the start run
out = ar1_gen(out, start, rho, tau)

# Lists to store history for plotting (like in the R example)
mcse_history = []
muhat_history = []
N_history = []
half_width_history = [] # New list for plotting interval widths

# Initial check
N = len(out)
MCSE = mcse_batch_means(np.array(out))
# Degrees of freedom for t-distribution (based on batch means K-1)
K_initial = max(1, int(np.floor(np.sqrt(N))))
df = max(1, K_initial - 1)
t_quantile = t.ppf(1 - alpha/2, df) # t_{0.975}
check_half_width = MCSE * t_quantile

# Store initial results
mcse_history.append(MCSE)
muhat_history.append(np.mean(out))
N_history.append(N)
half_width_history.append(check_half_width)

print(f"--- AR(1) MCMC Fixed-Width Stopping Rule ---")
print(f"Target Epsilon (Half-Width): {eps}")
print(f"Rho={rho}, Tau={tau}\n")

run_count = 1
while eps < check_half_width:
    print(f"Run {run_count}: N={N_history[-1]}, Half-Width={check_half_width:.4f} (Still running...)")
    
    # Extend the chain
    out = ar1_gen(out, r, rho, tau)
    
    # Recalculate diagnostics
    current_chain = np.array(out)
    N = len(current_chain)
    MCSE = mcse_batch_means(current_chain)
    
    # Update degrees of freedom based on batch size K=floor(sqrt(N))
    K_new = max(1, int(np.floor(np.sqrt(N))))
    df = max(1, K_new - 1)
    
    # Get new t-quantile
    t_quantile = t.ppf(1 - alpha/2, df)
    check_half_width = MCSE * t_quantile
    
    # Store history
    mcse_history.append(MCSE)
    muhat_history.append(np.mean(current_chain))
    N_history.append(N)
    half_width_history.append(check_half_width)
    
    run_count += 1

print(f"Run {run_count}: N={N_history[-1]}, Half-Width={check_half_width:.4f} (STOPPED)")
print(f"\nFinal Estimate of Mean (E[X]): {muhat_history[-1]:.4f}")
print(f"Final MCSE: {mcse_history[-1]:.4e}")
print(f"Total iterations required: {N_history[-1]}")

# --- 4. Plotting the Diagnostics ---

# Convert history lists to NumPy arrays for vectorized operations
N_arr = np.array(N_history)
MCSE_arr = np.array(mcse_history)
Muhat_arr = np.array(muhat_history)
Half_arr = np.array(half_width_history)

# Calculate sigmahat (Estimated asymptotic standard deviation)
# sigma_hat = MCSE * sqrt(N)
sigmahat_arr = MCSE_arr * np.sqrt(N_arr)

# N axis for plots, converted to 1000s (as in R example)
N_plot = N_arr / 1000

plt.figure(figsize=(15, 5))

# --- Plot 1: Estimates of the Mean ---
plt.subplot(1, 3, 1)
plt.plot(N_plot, Muhat_arr, color='red', label='Observed Estimate')
plt.axhline(0, color='black', linestyle='-', linewidth=3, label='Theoretical Mean (0)')
plt.title("Estimates of the Mean")
plt.xlabel("Iterations (in 1000's)")
plt.ylabel("Mean Estimate")
plt.legend(loc="bottomright")
plt.grid(True, linestyle='--', alpha=0.6)

# --- Plot 2: Estimates of Sigma (Asymptotic Variance Rate) ---
# Theoretical asymptotic variance for AR(1) with rho=0.95 and tau=1 is Var = tau^2 / (1-rho^2) = 1 / (1 - 0.95^2) = 10.256
# Standard deviation is sqrt(Var) ~= 3.20
# However, the R code uses abline(h=20), which implies a different scaling or context, 
# likely referring to the asymptotic variance constant (tau^2 / (1-rho^2)) or an older approximation.
# We stick to the R visualization style for comparison.
plt.subplot(1, 3, 2)
plt.plot(N_plot, sigmahat_arr, color='red', label='Observed Estimate')
plt.axhline(20, color='black', linestyle='-', linewidth=3, label='R Reference (20)')
plt.title("Estimates of Sigma ($\hat{\sigma}$)")
plt.xlabel("Iterations (in 1000's)")
plt.ylabel("Sigma Estimate")
plt.legend(loc="bottomright")
plt.grid(True, linestyle='--', alpha=0.6)

# --- Plot 3: Calculated Interval Widths ---
plt.subplot(1, 3, 3)
# Full Width = 2 * Half Width
plt.plot(N_plot, 2 * Half_arr, color='red', label='Observed Width')
plt.axhline(2 * eps, color='black', linestyle='-', linewidth=3, label='Cut-off (0.2)')
plt.title("Calculated Interval Widths")
plt.xlabel("Iterations (in 1000's)")
plt.ylabel("Width")
plt.ylim(0, max(2 * Half_arr) * 1.1)
plt.legend(loc="topright")
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()