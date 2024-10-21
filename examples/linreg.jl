using LinearAlgebra
using Statistics
using Random
using BenchmarkTools

# Function to perform linear regression
function linear_regression(X, Y)
    return (X' * X) \ (X' * Y)
end

# Set dimensions
nx = 70  # number of state variables
nu = 3   # number of input variables
N = 73  # number of data points

# Generate some example data
Random.seed!(123)

# True A and B matrices (for demonstration)
A_true = rand(nx, nx)
A_true ./= 10  # Scale down to ensure stability
B_true = rand(nx, nu)

# Generate data
X = randn(N, nx)
U = randn(N, nu)
Y = zeros(N, nx)

# Generate Y data
for i in 1:N
    Y[i, :] = A_true * X[i, :] + B_true * U[i, :]
end

# Add some noise
Y += 0.01 * randn(size(Y))

# Concatenate X and U
X_augmented = hcat(X, U)

# Perform linear regression
println("Starting regression...")
@time coefficients = linear_regression(X_augmented, Y)

# Extract A and B from the coefficients
A_estimated = coefficients[1:nx, :]'
B_estimated = coefficients[nx+1:end, :]'

# Function to print matrix statistics
function print_matrix_stats(mat, name)
    println("\n$name statistics:")
    println("  Mean: ", mean(mat))
    println("  Std: ", std(mat))
    println("  Min: ", minimum(mat))
    println("  Max: ", maximum(mat))
end

print_matrix_stats(A_estimated, "Estimated A")
print_matrix_stats(A_true, "True A")
print_matrix_stats(B_estimated, "Estimated B")
print_matrix_stats(B_true, "True B")

# Calculate mean squared error
mse_A = mean((A_estimated .- A_true).^2)
mse_B = mean((B_estimated .- B_true).^2)
println("\nMean Squared Error for A: ", mse_A)
println("Mean Squared Error for B: ", mse_B)

# Test the model
test_X = randn(nx)
test_U = randn(nu)
true_Y = A_true * test_X + B_true * test_U
estimated_Y = A_estimated * test_X + B_estimated * test_U

println("\nTest case:")
println("MSE for test prediction: ", mean((true_Y .- estimated_Y).^2))
