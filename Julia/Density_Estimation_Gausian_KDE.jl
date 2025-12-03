using Printf
using Distributions
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
            t = (data_point - x) / h
            S += (1/h) * kernel_func(t)
        end
        fhat[i] = (1/n_data) * S
    end
    return fhat
end

# --- Data Setup ---
eruption_data = generate_faithful_mock_data(200)
n = length(eruption_data)
h_kde = 0.1 # Smaller bandwidth for smoother estimate
x_range = collect(0.0:0.02:6.02) # Evaluation points (x)

# --- 1. Gaussian Kernel ---
function gaussian_kernel(t::Float64)::Float64
    """K(t) = 1/sqrt(2*pi) * exp(-t^2/2)"""
    # Use Julia's built-in constants
    return exp(-0.5 * t^2) / sqrt(2 * pi)
end

# --- Execution ---
fhat_gaussian = kernel_density_estimate(eruption_data, x_range, h_kde, gaussian_kernel)

@printf("--- Gaussian Kernel Density Estimator (h=%.1f) ---\n", h_kde)
@printf("This uses a smooth Gaussian kernel for continuous density estimation.\n")
@printf("%-10s | %-10s\n", "x", "Density")
@printf("-"^25 * "\n")

# Display a few key points
for i in 1:50:length(x_range)
    @printf("%-10.2f | %-10.4f\n", x_range[i], fhat_gaussian[i])
end