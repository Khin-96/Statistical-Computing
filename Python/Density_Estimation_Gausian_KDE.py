import numpy as np
from density_simulation import generate_faithful_mock_data
from naive_estimator_python import kernel_density_estimate # Reuse the generic KDE function

# --- Data Setup ---
eruption_data = generate_faithful_mock_data(n=200)
n = len(eruption_data)
h_kde = 0.1 # Smaller bandwidth for smoother estimate
x_range = np.arange(0.0, 6.02, 0.02) # Evaluation points (x)

# --- 1. Gaussian Kernel ---
def gaussian_kernel(t: float) -> float:
    """
    K(t) = 1/sqrt(2*pi) * exp(-t^2/2)
    This is the PDF of the standard normal distribution N(0, 1).
    """
    return np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)

# --- Execution ---
fhat_gaussian = kernel_density_estimate(eruption_data, x_range, h_kde, gaussian_kernel)

print(f"--- Gaussian Kernel Density Estimator (h={h_kde}) ---")
print("This uses a smooth Gaussian kernel for continuous density estimation.")
print(f"{'x':<10} | {'Density':<10}")
print("-" * 25)

# Display a few key points
for i in range(0, len(x_range), 50):
    print(f"{x_range[i]:<10.2f} | {fhat_gaussian[i]:<10.4f}")