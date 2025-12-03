using Distributions
using Random

"""
Generates a synthetic bimodal dataset to mimic Old Faithful eruption lengths.
It mixes two normal distributions: Short Eruptions and Long Eruptions.
"""
function generate_faithful_mock_data(n::Int=200, p_short::Float64=0.4)::Vector{Float64}
    n1 = round(Int, n * p_short)
    n2 = n - n1
    
    # Component 1 (Short): Mean 2.0, StdDev 0.3
    data1 = rand(Normal(2.0, 0.3), n1)
    # Component 2 (Long): Mean 4.5, StdDev 0.5
    data2 = rand(Normal(4.5, 0.5), n2)
    
    eruption_data = vcat(data1, data2)
    # Shuffle to mix the components naturally
    shuffle!(eruption_data)
    
    return eruption_data
end

# Generate and display the first few data points
eruption_data = generate_faithful_mock_data(200)
println("Generated $(length(eruption_data)) bimodal data points.")
println("First 5 data points: $(eruption_data[1:5])")