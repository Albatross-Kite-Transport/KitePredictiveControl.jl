using LinearAlgebra
using Statistics
using Random
using BenchmarkTools
using KiteModels

Ts = 0.05
if !@isdefined kite
    kite = KPS4_3L(KCU(se("system_3l.yaml")))
end
init_set_values = [-0.1, -0.1, -70.0]
init_sim!(kite; prn=true, torque_control=true, init_set_values)
x0 = copy(kite.integrator.u)

function linear_regression(X, Y)
    return (X' * X) \ (X' * Y)
end

function x_plus!(kite::KPS4_3L, x, u, Ts)
    OrdinaryDiffEqCore.reinit!(kite.integrator, x; t0=1.0, tf=1.0+Ts)
    next_step!(kite; set_values = u, dt = Ts)
    return kite.integrator.u
end

# Set dimensions
nx = 70  # number of state variables
nu = 3   # number of input variables
N = nx+nu  # number of data points

# Generate data
X = zeros(N, nx)
U = zeros(N, nu)
X_plus = zeros(N, nx)

# Generate Y data
for i in 1:N
    # Y[i, :] = A_true * X[i, :] + B_true * U[i, :]
    x_plus!(kite, x0, U[])
    X_plus[i, :] .= 
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
