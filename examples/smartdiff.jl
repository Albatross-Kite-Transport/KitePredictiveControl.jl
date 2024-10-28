"""
heading_y'   = turn_rate_y    +   0 * set_value
turn_rate_y' = k_1 * one      +   k_4 * set_value

flap_angle' = flap_vel          +   0 * set_value
flap_vel'   = acc = k_2 * one   +   k_5 * set_value

tether_length'  = tether_vel        +   0 * set_value
tether_vel'     = acc = k_3 * one   +   k_6 * set_value

one' = 0.0


X_prime = A * X + B * U

... * A = X_prime


c2d
"""

using ModelingToolkit, OrdinaryDiffEq, OrdinaryDiffEqCore, ControlSystems
import ModelingToolkit.SciMLBase: successful_retcode
using SymbolicIndexingInterface: getu, setp
using PreallocationTools
using SparseDiffTools
using SparseArrays
using LinearAlgebra
using KiteModels
using IterativeSolvers

include("../src/mtk_interface.jl")

Ts = 0.1
if !@isdefined kite
    kite = KPS4_3L(KCU(se("system_3l.yaml")))
end

init_set_values = [-0.1, -0.1, -70.0]
init_sim!(kite; prn=true, torque_control=true, init_set_values)
next_step!(kite; set_values = init_set_values, dt = 1.0)
x0 = copy(kite.integrator.u)
u0 = init_set_values
umin, umax = [-20, -20, -500], [0.0, 0.0, 0.0]
sys = kite.prob.f.sys
inputs = [sys.set_values[i] for i in 1:3]
outputs = vcat(
        sys.heading_y, 
        sys.turn_rate_y, 
        vcat(sys.flap_angle),
        vcat(sys.flap_vel),
        vcat(sys.tether_length),
        vcat(sys.tether_vel),
    )
get_y = getu(kite.integrator, outputs)
x_simple_0 = vcat(get_y(kite.integrator), 1)

solver = OrdinaryDiffEq.QNDF()
(f!, h!, simple_state, nu, _, nsx, ny) = generate_f_h(kite, inputs, outputs, solver, Ts)

function idx(symbol)
    return ModelingToolkit.variable_index(string.(simple_state), symbol)
end

A_pattern = zeros(nsx, nsx)
B_pattern = zeros(nsx, nu)

# --- heading ---
A_pattern[idx(sys.heading_y), idx(sys.turn_rate_y)] = 1.0
A_pattern[idx(sys.turn_rate_y), nsx] = 1.0
B_pattern[idx(sys.turn_rate_y), [1,2]] .= 1.0

# --- flap angle ---
for i in 1:2
    A_pattern[idx(sys.flap_angle[i]), idx(sys.flap_vel[i])] = 1.0
    A_pattern[idx(sys.flap_vel[i]), nsx] = 1.0
    B_pattern[idx(sys.flap_vel[i]), i] = 1.0
    B_pattern[idx(sys.flap_vel[i]), 3] = 1.0
end

# --- tether length ---
for i in 1:3
    A_pattern[idx(sys.tether_length[i]), idx(sys.tether_vel[i])] = 1.0
    A_pattern[idx(sys.tether_vel[i]), nsx] = 1.0
    B_pattern[idx(sys.tether_vel[i]), i] = 1.0
end
A_pattern = sparse(A_pattern)
B_pattern = sparse(B_pattern)

x_simple_plus = ones(nsx)
f!(x_simple_plus, x0, u0)

nm = 8 # number of measurements
U = zeros(nm, nu)
X = zeros(size(U, 1), nsx)
X_prime = zeros(size(U, 1), nsx)
set_value = 0.0
for i in 1:nm
    global set_value
    if (i%4 == 1) set_value += 5.0 end
    U[i, :] .= u0 - [i%4==1 ? set_value : 0.0, i%4==2 ? set_value : 0.0, i%4==3 ? set_value : 0.0]
    @show U[i, :]
    f!(x_simple_plus, x0, U[i, :])
    X[i, 1:nsx] .= x_simple_0
    X[i, end] = 1
    X_prime[i, 1:nsx] .= (x_simple_plus .- x_simple_0) ./ Ts
end

XU = [X U]
AB_pattern = [A_pattern B_pattern]
AB = copy(AB_pattern)

# Solve for each row of [A B]
for i in 1:nsx
    if !isempty(findnz(B[i, :])[1])
        nz_indices = findnz(AB_pattern[i, :])[1]
        Z = XU[:, nz_indices]
        sol = Z \ X_prime[:, i]
        AB[i, nz_indices] = sol
    end
end

# Extract A and B from AB
A = AB[:, 1:nsx]
B = AB[:, nsx+1:end]
C = I(nsx)
D = spzeros(nsx, nu)

css = ss(A, B, C, D)
dss = c2d(css, Ts)

u_test = u0 - [10.0, 0.0, 0.0]
ulin_heading = []
lin_heading = []
for i in 1:20
    # x_simple_plus = copy(x_simple_0)
    u_test = u0 - [i, -i, 0.0]
    f!(x_simple_plus, x0, u_test)
    lin_plus = dss.A * x_simple_0 + dss.B * u_test
    append!(ulin_heading, x_simple_plus[1])
    append!(lin_heading, lin_plus[1])
end
using Plots
plot()
plot!(ulin_heading)
display(plot!(lin_heading))

lin_plus = dss.A * x_simple_0 + dss.B * u_test
diff_ulin = norm(x_simple_plus .- x_simple_0)
diff_lin = norm(lin_plus .- x_simple_0)

println("diff_ulin: ", diff_ulin, "\ndiff_lin: ", diff_lin)


# # Update only the non-zero elements of A and B
# A.nzval .= nonzeros(A_new)
# B.nzval .= nonzeros(B_new)

