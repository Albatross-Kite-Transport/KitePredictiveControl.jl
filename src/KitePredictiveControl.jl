# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

module KitePredictiveControl

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
import KiteModels: SystemStructure

export linearize

include("utils.jl")
include("plotting.jl")

function sim!(s::SymbolicAWEModel, p::ModelParams)
    N = Int(round(time÷dt))
    vsm_interval = Int(round(vsm_dt÷dt))
    plot_interval = Int(round(plot_dt÷dt))
end

function linearize(s::SymbolicAWEModel)
    KiteModels.init_sim!(s; remake=false, adaptive=true)
    find_steady_state!(s)
    s.set_stabilize(s.integrator, true)
    p = ModelParams(s)
    @time linearize!(s, p, A, B, C, D)
    return p
end

function set_measured!(s::SymbolicAWEModel, p::ModelParams, x)
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
    
    # get wing_pos, rotate it by elevation and azimuth around the x and z axis
    wing_pos = R_t_w * [0, 0, distance]
    # wing_vel from elevation_vel and azimuth_vel
    wing_vel = R_t_w * [-elevation_vel, azimuth_vel, tether_vel[1]]
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
    set_state(integ, [wing_pos, wing_vel, Q_b_w, ω_b, tether_length, tether_vel])
    return nothing
end

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

function linearize!(s::SymbolicAWEModel, p::ModelParams, A, B, C, D)
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

end
