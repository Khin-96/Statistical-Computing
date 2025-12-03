import numpy as np
from density_simulation import generate_faithful_mock_data

# --- Data Setup ---
eruption_data = generate_faithful_mock_data(n=200)
n = len(eruption_data)
h = 0.5 # Bandwidth (half-width of the rectangular box)
x_range = np.arange(0.0, 6.02, 0.02) # Evaluation points (x)

# --- 1. Naive/Rectangular Kernel ---
def naive_kernel(t: float) -> float:
    """K(t) = 1/2 for |t| < 1, 0 otherwise"""
    if abs(t) < 1.0:
        return 0.5
    else:
        return 0.0

# --- 2. Generic Kernel Density Estimator (KDE) ---
def kernel_density_estimate(data, x_points, h, kernel_func):
    n_data = len(data)
    fhat = np.zeros(len(x_points))

    for idx, x in enumerate(x_points):
        S = 0.0
        for data_point in data:
            # Calculate t = (data_point - x) / h
            t = (data_point - x) / h
            # Sum K(t) * (1/h)
            S += (1/h) * kernel_func(t)
        # f_hat(x) = (1/n) * S
        fhat[idx] = (1/n_data) * S
    return fhat

# --- Execution ---
fhat_naive = kernel_density_estimate(eruption_data, x_range, h, naive_kernel)

print(f"--- Naive Density Estimator (h={h}) ---")
print("This uses a rectangular kernel to create a piecewise-constant density estimate.")
print(f"{'x':<10} | {'Density':<10}")
print("-" * 25)

# Display a few key points
for i in range(0, len(x_range), 50):
    print(f"{x_range[i]:<10.2f} | {fhat_naive[i]:<10.4f}")