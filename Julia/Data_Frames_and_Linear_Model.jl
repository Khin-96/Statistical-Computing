using DataFrames
using Random
using GLM
using StatsModels
using Statistics

# --- 1. Data Loading/Preparation (Simulated birthwt data) ---
Random.seed!(42)
N_rows = 189

# Data generation based on summary(birthwt)
data = DataFrame(
    low = rand(0:1, N_rows),
    age = rand(14:45, N_rows),
    lwt = rand(80:250, N_rows),
    race = rand([1, 2, 3], N_rows),
    smoke = rand(0:1, N_rows),
    bwt = rand(709:4990, N_rows)
)

# Manually inject the outlier for replication purposes:
data[1, :age] = 45
data[1, :bwt] = 4990

# Replicate R-style data prep
data.mother_smokes = [s == 1 ? "Yes" : "No" for s in data.smoke]
data.birthwt_grams = data.bwt

println("--- DataFrame Summary (Julia Equivalent) ---")
println(first(data, 5, cols=[:birthwt_grams, :age, :mother_smokes]))
println("\n" * "-"^50 * "\n")

# --- 2. Simple Linear Model: bwt ~ age ---
# The formula interface in GLM uses the @formula macro
linear_model_2 = lm(@formula(birthwt_grams ~ age), data)

println("--- Simple Linear Model: bwt ~ age (All Data) ---")
println(linear_model_2)
println("\n" * "-"^50 * "\n")


# --- 3. Outlier Removal and Simple Linear Model ---
# Remove the outlier row where age > 40
df_noout = data[data.age .<= 40, :]

linear_model_3 = lm(@formula(birthwt_grams ~ age), df_noout)

println("--- Simple Linear Model: bwt ~ age (Outlier Removed) ---")
println(linear_model_3)
println("\n" * "-"^50 * "\n")


# --- 4. Multiple Linear Model: bwt ~ smoke + age ---
# Julia automatically handles the categorical 'mother_smokes' variable
linear_model_3a = lm(@formula(birthwt_grams ~ mother_smokes + age), df_noout)

println("--- Multiple Linear Model: bwt ~ smoke + age (Outlier Removed) ---")
println(linear_model_3a)