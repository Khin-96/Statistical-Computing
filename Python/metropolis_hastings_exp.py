import numpy as np
import matplotlib.pyplot as plt

# --- Target Distribution (Exp(1)) ---
def target_pdf(x):
    """Unnormalized PDF of the Exponential(lambda=1) distribution."""
    if x < 0:
        return 0.0
    # pdf(x) = lambda * exp(-lambda * x). With lambda=1, it's exp(-x).
    return np.exp(-x)

# --- Proposal Distribution ---
# q(x'|x) = N(x, sigma^2). This is symmetric, so q(x|x') = q(x'|x).
# The ratio q(x|x') / q(x'|x) simplifies to 1 in the acceptance ratio.
def proposal_draw(current_x, step_size):
    """Draw a candidate sample x' from N(current_x, step_size^2)"""
    return np.random.normal(current_x, step_size)

# --- Metropolis-Hastings Sampler ---
def metropolis_hastings(target_pdf, proposal_draw, step_size, initial_x, n_samples):
    samples = np.zeros(n_samples)
    current_x = initial_x
    
    for i in range(n_samples):
        # 1. Propose a candidate x'
        proposed_x = proposal_draw(current_x, step_size)
        
        # 2. Compute acceptance ratio $\alpha$ (using unnormalized densities)
        # alpha = min(1, [pi(x') * q(x|x')] / [pi(x) * q(x'|x)])
        # For a symmetric proposal, q ratio is 1.
        
        # Log-acceptance ratio for numerical stability:
        # log_alpha = min(0, log(pi(x')) - log(pi(x)))
        log_alpha = np.log(target_pdf(proposed_x)) - np.log(target_pdf(current_x))
        alpha = np.exp(min(0, log_alpha)) # np.exp(0) = 1
        
        # 3. Accept or Reject
        if np.random.rand() < alpha:
            current_x = proposed_x # Accept
            
        samples[i] = current_x # Store the current state (accepted or rejected)
        
    return samples

# --- Run Simulation ---
N = 50000        # Number of samples
BURN_IN = 1000   # Number of samples to discard
STEP_SIZE = 1.0  # Proposal standard deviation (tuning parameter)

# Since Exp(1) is only defined for x >= 0, we choose a non-negative start.
initial_x = 2.0 

print(f"Running MH for Exp(1) with N={N} and step_size={STEP_SIZE}...")
mh_samples = metropolis_hastings(target_pdf, proposal_draw, STEP_SIZE, initial_x, N)
final_samples = mh_samples[BURN_IN:]

# --- Plotting Results ---
x = np.linspace(0, 8, 1000)
# True Exp(1) PDF: f(x) = 1 * exp(-1 * x)
true_pdf = np.exp(-x) 

plt.figure(figsize=(10, 6))
plt.hist(final_samples, bins=50, density=True, label='MH Samples (Normalized Histogram)', alpha=0.7)
plt.plot(x, true_pdf, 'r-', linewidth=2, label='True Exp(1) PDF')
plt.title(f'Metropolis-Hastings Sampler for Exp(1) (Step Size $\sigma$={STEP_SIZE})')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 8)
plt.show()

# Print mean for comparison (True mean of Exp(1) is 1.0)
print(f"Sample Mean: {np.mean(final_samples):.4f}")
print(f"Acceptance Rate: {np.sum(np.diff(mh_samples) != 0) / (N - 1):.4f}")