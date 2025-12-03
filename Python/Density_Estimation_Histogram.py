import numpy as np
from density_simulation import generate_faithful_mock_data

# --- Data Setup ---
eruption_data = generate_faithful_mock_data(n=200)
x0, x1 = 0, 8
h = 0.5 # Bin width

# --- Histogram Calculation ---
# 1. Define breaks (bins)
my_breaks = np.arange(x0, x1 + h, h)

# 2. Use numpy.histogram to count frequencies and calculate density
# density=True normalizes the counts so the area sums to 1.
counts, bin_edges = np.histogram(
    eruption_data, 
    bins=my_breaks, 
    density=True
)

# Bin mids are (edge[i] + edge[i+1]) / 2
mids = (bin_edges[:-1] + bin_edges[1:]) / 2

print(f"--- Histogram Density Estimate (Bin Width h={h}) ---")
print("This gives a piecewise-constant estimate of the density.")
print(f"Total Bins: {len(mids)}\n")
print(f"{'Midpoint':<10} | {'Density':<10} | {'Bin Edges':<20}")
print("-" * 45)
for m, d, e_start, e_end in zip(mids, counts, bin_edges[:-1], bin_edges[1:]):
    print(f"{m:<10.2f} | {d:<10.4f} | [{e_start:.1f}, {e_end:.1f})")

# The density is constant (d) within each bin [e_start, e_end).