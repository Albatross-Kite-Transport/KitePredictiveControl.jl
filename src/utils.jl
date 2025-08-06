# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

const Setter = SymbolicIndexingInterface.MultipleSetters
const Getter = SymbolicIndexingInterface.AbstractStateGetIndexer
const PGetter = SymbolicIndexingInterface.AbstractParameterGetIndexer

struct ModelParams
    s::SymbolicAWEModel
    integ::OrdinaryDiffEq.ODEIntegrator
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Matrix{Float64}
    D::Matrix{Float64}
    set_x::Setter
    set_ix::Setter
    get_x::Getter
    get_y::Getter
    set_u::Setter
    get_u::PGetter
    set_measured_x::Setter
    get_measured_x::Getter
    get_lin_x::Getter
    get_lin_dx::Getter
    get_lin_y::Getter
    get_distance::Getter
    dt::Float64
    x_vec::Vector{Num}
    y_vec::Vector{Num}
    u_vec::Vector{Num}
    lin_x_vec::Vector{Num}
    lin_dx_vec::Vector{Num}
    lin_y_vec::Vector{Num}
    x0::Vector{Float64}
    y0::Vector{Float64}
    u0::Vector{Float64}
    x_idxs::Dict{Num, Int64}
    y_idxs::Dict{Num, Int64}
    Q_idxs::Vector{Int64}
    function ModelParams(s, x0=nothing, y0=nothing, u0=nothing)
        sys = s.sys
        x_vec = ModelingToolkit.unknowns(sys)
        u_vec = [sys.set_values[i] for i in 1:3]
        y_vec = [
            sys.elevation[1]
            sys.elevation_vel[1]
            sys.azimuth[1]
            sys.azimuth_vel[1]
            sys.heading[1]
            sys.turn_rate[1,3]
            sys.tether_length[1]
            sys.tether_length[2]
            sys.tether_length[3]
            sys.tether_vel[1]
            sys.tether_vel[2]
            sys.tether_vel[3]
        ]
        lin_x_vec = [
            sys.heading[1]
            sys.turn_rate[1,3]
            sys.tether_length[1]
            sys.tether_length[2]
            sys.tether_length[3]
            sys.tether_vel[1]
            sys.tether_vel[2]
            sys.tether_vel[3]
        ]
        lin_dx_vec = [
            sys.turn_rate[1,3]
            sys.turn_acc[1,3]
            sys.tether_vel[1]
            sys.tether_vel[2]
            sys.tether_vel[3]
            sys.tether_acc[1]
            sys.tether_acc[2]
            sys.tether_acc[3]
        ]
        lin_y_vec = [
            sys.heading[1]
            sys.tether_length[1]
            sys.angle_of_attack[1]
            sys.winch_force[1]
        ]
        measured_x_vec = [
            sys.wing_pos[1,:], 
            sys.wing_vel[1,:], 
            sys.Q_b_w[1,:], 
            sys.ω_b[1,:], 
            sys.tether_length, 
            sys.tether_vel
        ]
        nx = length(x_vec)
        ny = length(y_vec)
        nu = length(u_vec)
        A = zeros(nx, nx)
        B = zeros(nx, nu)
        C = zeros(ny, nx)
        D = zeros(ny, nu)

        set_x = setu(s.integrator, x_vec)
        set_ix = setu(s.integrator, Initial.(x_vec))
        get_x = getu(s.integrator, x_vec)
        get_y = getu(s.integrator, y_vec)
        set_u = setu(s.integrator, u_vec)
        get_u = getu(s.integrator, u_vec)
        set_measured_x = setu(s.integrator, measured_x_vec)
        get_measured_x = getu(s.integrator, measured_x_vec)
        get_lin_x = getu(s.integrator, lin_x_vec)
        get_lin_dx = getu(s.integrator, lin_dx_vec)
        get_lin_y = getu(s.integrator, lin_y_vec)
        get_distance = getu(s.integrator, sys.distance)
        
        isnothing(x0) && (x0 = get_x(s.integrator))
        isnothing(y0) && (y0 = get_y(s.integrator))
        isnothing(u0) && (u0 = get_u(s.integrator))
        x_idxs = Dict{Num, Int}()
        for (idx, sym) in enumerate(x_vec)
            x_idxs[sym] = idx
        end
        y_idxs = Dict{Num, Int}()
        for (idx, sym) in enumerate(y_vec)
            y_idxs[sym] = idx
        end
        Q_idxs = [x_idxs[sys.Q_b_w[1,i]] for i in 1:4]
        
        return new(
            s, s.integrator,
            A, B, C, D,
            set_x, set_ix, get_x, 
            get_y, set_u, get_u,
            set_measured_x, get_measured_x,
            get_lin_x, get_lin_dx, get_lin_y, get_distance,
            1/s.set.sample_freq, x_vec, y_vec, u_vec,
            lin_x_vec, lin_dx_vec, lin_y_vec,
            x0, y0, u0,
            x_idxs, y_idxs, Q_idxs
        )
    end
end

