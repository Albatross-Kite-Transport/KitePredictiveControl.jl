# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

"""
Use a nonlinear MTK model with online linearization and a linear MPC using ModelPredictiveControl.jl
"""

using Plots
using KiteModels, KiteUtils, LinearAlgebra
using ModelPredictiveControl, ModelingToolkit, OrdinaryDiffEq, DifferentiationInterface,
    SymbolicIndexingInterface
using ModelingToolkit: setu, setp, getu, getp, @variables
using ModelingToolkit: t_nounits as t
using KiteModels: rotate_around_y, rotate_around_z, rotation_matrix_to_quaternion
using SciMLBase: successful_retcode
using RobustAndOptimalControl
using ControlSystems
using Printf

include("utils.jl")
include("plotting.jl")
set_data_path(joinpath(dirname(@__DIR__), "data"))

ad_type = AutoForwardDiff()
dt = 0.1
time = 4.8
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
s_model = SymbolicAWEModel(set_model)
s_plant = SymbolicAWEModel(set_plant)

measure = Measurement()
measure.sphere_pos .= deg2rad.([70.0 70.0; 1.0 -1.0])
KiteModels.init_sim!(s_plant, measure; remake=false, adaptive=true)
sys = s_plant.sys

measured_states = @variables begin
    elevation(t)
    elevation_vel(t)
    azimuth(t)
    azimuth_vel(t)
    heading(t)
    turn_rate(t)[3]
    tether_length(t)[1:3]
    tether_vel(t)[1:3]
end
measured_states = reduce(vcat, Symbolics.scalarize.(measured_states))
unknowns = KiteModels.get_nonstiff_unknowns(s_plant)
lin_outputs = unique([
    measured_states
    unknowns
])
KiteModels.init_sim!(s_model, measure; remake=false, adaptive=true, lin_outputs)

function stabilize!(s, T=10, lin_vsm=true)
    s.integrator.ps[s.sys.stabilize] = true
    for _ in 1:T÷dt
        lin_vsm && KiteModels.linearize_vsm!(s)
        OrdinaryDiffEq.step!(s.integrator, dt)
        @assert successful_retcode(s.integrator.sol)
    end
    s.integrator.ps[s.sys.stabilize] = false
    nothing
end
stabilize!(s_plant)

function set_y!(p::ModelParams, y)
    # get variables from y
    elevation       = y[1]
    elevation_vel   = y[2]
    azimuth         = y[3]
    azimuth_vel     = y[4]
    heading         = y[5]
    turn_rate       = y[6]
    tether_length   = y[7:9]
    tether_vel      = y[10:12]

    # get variables from integrator
    distance = p.get_state(p.integ)
    R_t_w = KiteModels.calc_R_t_w(elevation, azimuth) # rotation of tether to world, similar to view rotation, but always pointing up
    
    # get kite_pos, rotate it by elevation and azimuth around the x and z axis
    kite_pos = R_t_w * [0, 0, distance]
    # kite_vel from elevation_vel and azimuth_vel
    kite_vel = R_t_w * [-elevation_vel, azimuth_vel, tether_vel[1]]
    # find quaternion orientation from heading, R_cad_body and R_t_w
    x = [cos(-heading), -sin(-heading), 0]
    y = [sin(-heading),  cos(-heading), 0]
    z = [0, 0, 1]
    R_b_w = R_t_w * p.s.wing.R_cad_body' * [x y z]
    Q_b_w = rotation_matrix_to_quaternion(R_b_w)
    # adjust the turn rates for observed turn rate
    ω_b = R_b_w' * R_t_w * [0, 0, turn_rate]
    # directly set tether length
    # directly set tether vel
    p.set_state(p.integ, [kite_pos, kite_vel, Q_b_w, ω_b, tether_length, tether_vel])
    return nothing
end

# TODO!! CUSTOM DYNAMICSS https://github.com/SciML/SteadyStateDiffEq.jl/blob/201202ae468f5e2cc1d0e2229e3737a9dfb086c1/src/solve.jl
function ModelPredictiveControl.preparestate!(p::ModelParams, y)
    set_y!(p, y)
    OrdinaryDiffEq.reinit!(p.integ, p.integ.u; reinit_dae=false)
    stabilize!(p.s, dt, false)
    return p.get_x(p.integ)
end

function ModelPredictiveControl.updatestate!(p::ModelParams, u, y)
    # set_y!(p, y)
    # p.set_u(p.integ, u)
    # OrdinaryDiffEq.reinit!(p.integ,    p.integ.u; reinit_dae=false)
    # OrdinaryDiffEq.step!(p.integ, dt)
    # @assert successful_retcode(p.integ.sol)
    # stabilize!(p.s, 0.1)
    return p.get_x(p.integ)
end


# https://github.com/SciML/ModelingToolkit.jl/issues/3552#issuecomment-2817642041
function f(x, u, _, p)
    p.set_x(p.integ, x)
    p.set_u(p.integ, u)
    OrdinaryDiffEq.reinit!(p.integ, p.integ.u; reinit_dae=false)
    OrdinaryDiffEq.step!(p.integ, dt)
    @assert successful_retcode(p.integ.sol)
    xnext = p.get_x(p.integ)
    return xnext
end

function h(x, _, p)
    p.set_x(p.integ, x)
    y = p.get_y(p.integ)
    return y
end

p_plant = ModelParams(s_plant, lin_outputs)
p_model = ModelParams(s_model, lin_outputs, p_plant.x0, p_plant.y0, p_plant.u0)

nu, nx, ny = length(p_plant.u_vec), length(p_plant.x_vec), length(p_plant.y_vec)
vx, vu, vy = string.(p_plant.x_vec), string.(p_plant.u_vec), string.(p_plant.y_vec)
plant = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
# setstate!(plant, p_plant.x0)
# x = preparestate!(p_model, plant())

# TODO: linearize on a different core https://www.perplexity.ai/search/using-a-julia-scheduler-run-tw-oKloXmWmSR6YWb47nW_1Gg#0
function linearize!(p::ModelParams, linmodel, x, u, y, lin_vsm=true)
    vsm_t = @elapsed begin
        p.set_x(p.integ, x)
        OrdinaryDiffEq.reinit!(p.integ, p.integ.u; reinit_dae=false)
        lin_vsm && KiteModels.linearize_vsm!(p.s)
    end
    p.x0 .= x
    p.u0 .= u
    p.y0 .= y
    lin_t = @elapsed begin
        (; A, B, C, D) = KiteModels.linearize(p.s)
        csys = ss(A, B, C, D)
        if trunc
            tsys, hs, _ = baltrunc_unstab(csys; residual=true, n=length(unknowns))
        else
            tsys = csys
        end
        dsys = c2d(tsys, dt, :zoh)
    end
    linmodel.A .= dsys.A
    linmodel.Bu .= dsys.B
    linmodel.C .= dsys.C
    setop!(linmodel, uop=p_model.u0, yop=p_model.y0)
    linmodel.xop .= 0.0
    return vsm_t, lin_t
end

KiteModels.linearize_vsm!(p_model.s)
KiteModels.linearize_vsm!(p_plant.s)

i_nonstiff = [p_model.y_idxs[sym] for sym in unknowns]
i_ym = [p_model.y_idxs[sym] for sym in measured_states]
linmodel = ModelPredictiveControl.LinModel(zeros(nx,nx), zeros(nx,nu), zeros(ny,nx), zeros(nx,0), zeros(ny,0), dt)
setname!(linmodel; u=vu, y=vy)

Hp, Hc, Mwt = 100, 2, fill(0.0, linmodel.ny)
Nwt = [100.0, 10.0, 10.0]
Mwt[p_model.y_idxs[sys.tether_length[1]]] = 10.0
# Mwt[p_model.y_idxs[sys.tether_length[2]]] = 1.0
# Mwt[p_model.y_idxs[sys.tether_length[3]]] = 1.0
Mwt[p_model.y_idxs[sys.free_twist_angle[2]]] = 1.0
Mwt[p_model.y_idxs[sys.free_twist_angle[3]]] = 1.0
Mwt[p_model.y_idxs[sys.kite_pos[2]]] = 1.0

umin, umax = [-100, -20, -20], [0, 0, 0]
man = ManualEstimator(linmodel; i_ym, nint_u=0, nint_ym=0)
mpc = LinMPC(man; Hp, Hc, Mwt, Nwt, Cwt=Inf)
mpc = setconstraint!(mpc; umin, umax)

function sim_adapt!(mpc, linmodel, N, ry, plant, x0, px0, x̂0, y_step=zeros(plant.ny))
    U_data, Y_data, Ry_data, X_data, X̂_data, Ŷ_data = 
        zeros(linmodel.nu, N), zeros(linmodel.ny, N), zeros(linmodel.ny, N), zeros(linmodel.nx, N), zeros(linmodel.nx, N), zeros(linmodel.ny, N)
    setstate!(plant, px0)
    initstate!(mpc, p_model.u0, plant()[i_ym])

    x = preparestate!(p_model, plant())
    linearize!(p_model, linmodel, x, p_model.u0, plant())
    setmodel!(mpc, linmodel)

    println("\nStarting simulation...")
    println("─"^80)
    println(
        rpad("Time", 12),
        rpad("Real-time", 12),
        rpad("VSM", 12),
        rpad("Lin", 12),
        rpad("MPC", 12),
        rpad("Estim", 12),
        "Norm(A)"
    )
    println("─"^80)
    for i = 1:N
        t = @elapsed begin
            y = plant() + y_step
            estim_t = @elapsed x̂ = preparestate!(p_model, y[i_ym])
            setstate!(mpc, x̂ .- p_model.x0)
            mpc_t = @elapsed u = moveinput!(mpc, ry)

            vsm_t, lin_t = Inf, Inf
            lin_vsm = i%vsm_interval==0
            vsm_t, lin_t = linearize!(p_model, linmodel, x̂, u, y, lin_vsm)
            setmodel!(mpc, linmodel)

            U_data[:,i], Y_data[:,i], Ry_data[:,i], X_data[:,i], X̂_data[:,i], Ŷ_data[:,i] = u, y, ry, p_plant.integ.u, x̂[1:length(x0)], mpc.ŷ
            estim_t += @elapsed x̂ = updatestate!(p_model, u, y[i_ym])
            setstate!(mpc, x̂ .- p_model.x0)
        end

        KiteModels.linearize_vsm!(p_plant.s)
        updatestate!(plant, u)

        if i%plot_interval==0
            display(plot(p_model.s, kite_zoom=true))
        end

        @printf("%8.2fs │ %8.3fx │ %8.3fx │ %8.3fx │ %8.1fx │ %8.1fx │ %.2e\n",
            i*dt, dt/t, dt/vsm_t, dt/lin_t, dt/mpc_t, dt/estim_t, norm(linmodel.A))
    end
    res = SimResult(mpc, U_data, Y_data; Ry_data, X_data, X̂_data, Ŷ_data)
    return res
end

ry = copy(p_model.y0)
# ry[p_model.y_idxs[sys.kite_pos[2]]] = 10.0
# ry[p_model.y_idxs[sys.angle_of_attack]] = deg2rad(10)

x0 = p_model.get_x(p_model.integ)
x̂0 = x0
# x̂0 = zeros(estim.nx̂)
res = sim_adapt!(mpc, linmodel, N, ry, plant, x0, p_plant.x0, x̂0)
y_idxs = [
    findall(x -> x != 0.0, Mwt)
    # [p_model.y_idxs[sys.Q_b_w[i]] for i in 1:4]
    # p_model.y_idxs[sys.kite_pos[3]]
    p_model.y_idxs[sys.ω_b[3]]
    # p_model.y_idxs[sys.free_twist_angle[1]]
]
Plots.plot(res; plotx=[34], ploty=y_idxs, 
    plotxwithx̂=[34],
    plotŷ=true,
    plotu=true, size=(900, 900)
)
