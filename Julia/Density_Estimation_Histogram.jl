using Statistics
using Printf
# Note: In a real environment, you would use the definition from the previous file.
# We'll re-define it here for a self-contained runnable file.
function generate_faithful_mock_data(n::Int=200, p_short::Float64=0.4)::Vector{Float64}
    using Distributions, Random
    n1 = round(Int, n * p_short)
    n2 = n - n1
    data1 = rand(Normal(2.0, 0.3), n1)
    data2 = rand(Normal(4.5, 0.5), n2)
    eruption_data = vcat(data1, data2)
    shuffle!(eruption_data)
    return eruption_data
end

# --- Data Setup ---
eruption_data = generate_faithful_mock_data(200)
x0, x1 = 0.0, 8.0
h = 0.5 # Bin width

# --- Histogram Calculation ---
# 1. Define breaks (bins)
my_breaks = collect(x0:h:x1)

# 2. Calculate the histogram using StatsBase.Histogram
# Since we can't easily import StatsBase here, we implement the logic directly
# to match the R/Python behavior (density normalization).

# Simple Binning Logic (Manual Density Calculation)
bin_counts = zeros(Int, length(my_breaks) - 1)
bin_edges = my_breaks
n_data = length(eruption_data)
bin_width = h

for x_val in eruption_data
    for i in 1:(length(bin_edges) - 1)
        # Check if x_val is in [bin_edges[i], bin_edges[i+1])
        if x_val >= bin_edges[i] && (x_val < bin_edges[i+1] || (x_val == x1 && i == length(bin_edges) - 1))
            bin_counts[i] += 1
            break
        end
    end
end

# Density = (Count / n) / Bin Width
density_values = (bin_counts ./ n_data) ./ bin_width
mids = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in 1:(length(bin_edges) - 1)]

@printf("--- Histogram Density Estimate (Bin Width h=%.1f) ---\n", h)
@printf("This gives a piecewise-constant estimate of the density.\n")
@printf("Total Bins: %d\n\n", length(mids))
@printf("%-10s | %-10s | %-20s\n", "Midpoint", "Density", "Bin Edges")
@printf("-"^45 * "\n")
for i in 1:length(mids)
    @printf("%-10.2f | %-10.4f | [%.1f, %.1f)\n", mids[i], density_values[i], bin_edges[i], bin_edges[i+1])
end