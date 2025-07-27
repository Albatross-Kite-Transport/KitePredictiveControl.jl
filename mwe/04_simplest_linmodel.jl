# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

"""
Simplify the nonlinear model to a very simple linear model
States:
    - tether length vel acc
    - heading, turn rate, turn acc
Inputs
    - set torque
Outputs
    - heading
    - angle of attack (linear combination of tether lengths)

Sparsity pattern:
    - Torques to tether acc
    - Tether length to turn acc
"""

using Plots
using KiteModels, KiteUtils, LinearAlgebra
using ModelPredictiveControl, ModelingToolkit, OrdinaryDiffEq,
    SymbolicIndexingInterface
using ModelingToolkit: setu, setp, getu, getp, @variables
using ModelingToolkit: t_nounits as t
using KiteModels: rotation_matrix_to_quaternion
using SciMLBase: successful_retcode
using RobustAndOptimalControl
using ControlSystems
using Printf

include("utils.jl")
include("plotting.jl")
set_data_path(joinpath(dirname(@__DIR__), "data"))

dt = 0.05
time = 10.0
vsm_dt = 0.1
plot_dt = 0.1
trunc = false

N = Int(round(time÷dt))
vsm_interval = Int(round(vsm_dt÷dt))
plot_interval = Int(round(plot_dt÷dt))

# Initialize model
set_model = deepcopy(load_settings("system_model.yaml"))
set_plant = deepcopy(load_settings("system_plant.yaml"))
set_model.quasi_static = false
set_plant.quasi_static = false
ram_model = SymbolicAWEModel(set_model)

KiteModels.init_sim!(ram_model; remake=false, adaptive=true)
find_steady_state!(ram_model)
sys = ram_model.sys
integ = ram_model.integrator
ram_model.integrator.ps[sys.stabilize] = true

y_vec = [
    sys.elevation
    sys.elevation_vel
    sys.azimuth
    sys.azimuth_vel
    sys.heading
    sys.turn_rate[3]
    sys.tether_length[1]
    sys.tether_length[2]
    sys.tether_length[3]
    sys.tether_vel[1]
    sys.tether_vel[2]
    sys.tether_vel[3]
]
lin_x_vec = [
    sys.heading
    sys.turn_rate[3]
    sys.tether_length[1]
    sys.tether_length[2]
    sys.tether_length[3]
    sys.tether_vel[1]
    sys.tether_vel[2]
    sys.tether_vel[3]
]
lin_dx_vec = [
    sys.turn_rate[3]
    sys.turn_acc[3]
    sys.tether_vel[1]
    sys.tether_vel[2]
    sys.tether_vel[3]
    sys.tether_acc[1]
    sys.tether_acc[2]
    sys.tether_acc[3]
]
lin_y_vec = [
    sys.heading
    sys.tether_length[1]
    sys.angle_of_attack
    sys.winch_force[1]
]

nx = length(lin_x_vec)
ny = length(lin_y_vec)

c = collect
get_distance = getu(integ, sys.distance)
set_state = setu(integ, 
    [sys.kite_pos, sys.kite_vel, sys.Q_b_w, sys.ω_b, sys.tether_length, sys.tether_vel]
)

"""
Calculate τ such that tether_acc = 0
"""
function calc_steady_τ(wm::TorqueControlledMachine, winch_force, tether_vel)
    ω = wm.set.gear_ratio/wm.set.drum_radius * tether_vel
    τ_friction = WinchModels.calc_coulomb_friction(wm) * WinchModels.smooth_sign(ω) + WinchModels.calc_viscous_friction(wm, ω)
    steady_τ = -wm.set.drum_radius / wm.set.gear_ratio * winch_force  + τ_friction
    return steady_τ
end

function set_measured!(s::SymbolicAWEModel, x)
    # get variables from y
    elevation       = x[1]
    elevation_vel   = x[2]
    azimuth         = x[3]
    azimuth_vel     = x[4]
    heading         = x[5]
    turn_rate       = x[6]
    tether_length   = x[7:9]
    tether_vel      = x[10:12]

    # get variables from integrator
    distance = get_distance(integ)
    R_t_w = KiteModels.calc_R_t_w(elevation, azimuth) # rotation of tether to world, similar to view rotation, but always pointing up
    
    # get kite_pos, rotate it by elevation and azimuth around the x and z axis
    kite_pos = R_t_w * [0, 0, distance]
    # kite_vel from elevation_vel and azimuth_vel
    kite_vel = R_t_w * [-elevation_vel, azimuth_vel, tether_vel[1]]
    # find quaternion orientation from heading, R_cad_body and R_t_w
    x = [cos(-heading), -sin(-heading), 0]
    y = [sin(-heading),  cos(-heading), 0]
    z = [0, 0, 1]
    R_b_w = R_t_w * s.wing.R_cad_body' * [x y z]
    Q_b_w = rotation_matrix_to_quaternion(R_b_w)
    # adjust the turn rates for observed turn rate
    ω_b = R_b_w' * R_t_w * [0, 0, turn_rate]
    # directly set tether length
    # directly set tether vel
    set_state(integ, [kite_pos, kite_vel, Q_b_w, ω_b, tether_length, tether_vel])
    return nothing
end

integ = ram_model.integrator
get_lin_dx = getu(integ, lin_dx_vec)
get_lin_y = getu(integ, lin_y_vec)
get_sphere_pos_vel = getu(integ, y_vec[1:4])
set_u = setu(integ, sys.set_values)
get_u = getu(integ, sys.set_values)
get_y = getu(integ, y_vec)
set_x = setu(integ, unknowns(sys))
get_x = getu(integ, unknowns(sys))
x0 = get_x(integ)

function jacobian(f::Function, x::AbstractVector, ϵ::AbstractVector)
    n = length(x)
    fx = f(x)
    m = length(fx)
    J = zeros(m, n)
    for i in 1:n
        x_perturbed = copy(x)
        x_perturbed[i] += ϵ[i]
        J[:, i] = (f(x_perturbed) - fx) / ϵ[i]
    end
    return J
end

function linearize!(s::SymbolicAWEModel, A, B, C, D)
    integ = s.integrator
    lin_x0 = get_y(integ)[5:12]
    lin_u0 = get_u(integ)
    KiteModels.linearize_vsm!(s)
    A .= 0.0
    B .= 0.0
    C .= 0.0
    D .= 0.0
    
    function f(x, u)
        set_x(integ, x0)
        sphere_pos_vel = get_sphere_pos_vel(integ)
        set_measured!(s, [sphere_pos_vel; x])
        set_u(integ, u)
        OrdinaryDiffEq.reinit!(integ, integ.u; reinit_dae=false)
        OrdinaryDiffEq.step!(integ, dt)
        return get_lin_dx(integ)
    end

    # yes it looks weird to step in an output function, but this is a steady state finder rather than output
    function h(x)
        set_x(integ, x0)
        sphere_pos_vel = get_sphere_pos_vel(integ)
        set_measured!(s, [sphere_pos_vel; x])
        OrdinaryDiffEq.reinit!(integ, integ.u; reinit_dae=false)
        OrdinaryDiffEq.step!(integ, dt)
        return get_lin_y(integ)
    end

    f_x(x) = f(x, lin_u0)
    f_u(u) = f(lin_x0, u)

    # calculate jacobian
    ϵ_x = [0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ϵ_u = [1.0, 0.1, 0.1]
    A .= jacobian(f_x, lin_x0, ϵ_x)
    B .= jacobian(f_u, lin_u0, ϵ_u)
    C .= jacobian(h,   lin_x0, ϵ_x)

    nothing
end

A = zeros(nx, nx)
B = zeros(nx, 3)
C = zeros(ny, nx)
D = zeros(ny, 3)
@time linearize!(ram_model, A, B, C, D)
@time linearize!(ram_model, A, B, C, D)

linsys = ss(A,B,C,D)
nothing


# julia> A
# 8×8 Matrix{Float64}:
#     0.000814894    1.00014         2.13609e-6      4.23583e-5    -4.27542e-5      2.37988e-7      3.76856e-10     1.79523e-11
#  -718.555        -53.0095         -8.8696         48.8487       -57.6373          0.0644731       1.92147e-6      9.18072e-8
#     0.0            0.0             0.0             0.0            0.0             1.0             0.0             0.0
#     0.0            0.0             0.0             0.0            0.0             0.0             1.0             0.0
#     4.44089e-14    6.93889e-15    -1.80411e-14    -4.57967e-15   -2.91434e-15     1.66533e-15    -8.32667e-16     1.0
#   -17.4877        -0.225774     -799.764          10.9828        10.644        -105.302           7.31719e-6      3.49831e-7
#   -50.9999         3.85377        19.3347       -100.288         20.0782          1.34941      -107.57            1.69535e-8
#    55.2985        -3.93163        21.6658         20.3021       -99.8842          1.35809         2.62089e-7   -101.663
