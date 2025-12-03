using Printf
# Include helper functions for self-contained runnable code
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
n = length(eruption_data)
h = 0.5 # Bandwidth (half-width of the rectangular box)
x_range = collect(0.0:0.02:6.02) # Evaluation points (x)

# --- 1. Naive/Rectangular Kernel ---
function naive_kernel(t::Float64)::Float64
    """K(t) = 1/2 for |t| < 1, 0 otherwise"""
    if abs(t) < 1.0
        return 0.5
    else
        return 0.0
    end
end

# --- 2. Generic Kernel Density Estimator (KDE) ---
function kernel_density_estimate(
    data::Vector{Float64},
    x_points::Vector{Float64},
    h::Float64,
    kernel_func::Function
)::Vector{Float64}
    n_data = length(data)
    fhat = zeros(length(x_points))

    for i in 1:length(x_points)
        x = x_points[i]
        S = 0.0
        for data_point in data
            # Calculate t = (data_point - x) / h
            t = (data_point - x) / h
            # Sum K(t) * (1/h)
            S += (1/h) * kernel_func(t)
        end
        # f_hat(x) = (1/n) * S
        fhat[i] = (1/n_data) * S
    end
    return fhat
end

# --- Execution ---
fhat_naive = kernel_density_estimate(eruption_data, x_range, h, naive_kernel)

@printf("--- Naive Density Estimator (h=%.1f) ---\n", h)
@printf("This uses a rectangular kernel to create a piecewise-constant density estimate.\n")
@printf("%-10s | %-10s\n", "x", "Density")
@printf("-"^25 * "\n")

# Display a few key points
for i in 1:50:length(x_range)
    @printf("%-10.2f | %-10.4f\n", x_range[i], fhat_naive[i])
end