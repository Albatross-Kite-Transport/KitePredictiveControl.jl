"""
Use a nonlinear MTK model with online linearization and a linear MPC using ModelPredictiveControl.jl
"""

using KiteModels, KiteUtils, LinearAlgebra
using ModelPredictiveControl, ModelingToolkit, Plots, OrdinaryDiffEq, DifferentiationInterface,
    SymbolicIndexingInterface
using ModelingToolkit: setu, setp, getu, getp, @variables
using ModelingToolkit: t_nounits as t
using KiteModels: rotate_around_y, rotate_around_z, rotation_matrix_to_quaternion
using SciMLBase: successful_retcode
using RobustAndOptimalControl
using ControlSystems
using Plots
using Printf

include("utils.jl")
include("plotting.jl")
set_data_path(joinpath(dirname(@__DIR__), "data"))

ad_type = AutoForwardDiff()
N = 10
trunc = false

# Initialize model
set_model = deepcopy(load_settings("system_model.yaml"))
set_plant = deepcopy(load_settings("system_plant.yaml"))
set_model.quasi_static = false
set_plant.quasi_static = false
s_model = RamAirKite(set_model)
s_plant = RamAirKite(set_plant)
dt = 1/set_model.sample_freq

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
    turn_rate(t)[1]
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

function stabilize!(s, T=10)
    s.integrator.ps[s.sys.stabilize] = true
    for _ in 1:T÷dt
        KiteModels.linearize_vsm!(s)
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

    @show tether_length

    # get variables from integrator
    distance = p.get_state(p.integ)
    R_t_w = KiteModels.calc_R_t_w(elevation, azimuth) # rotation of tether to world, similar to view rotation, but always pointing up
    
    # get kite_pos, rotate it by elevation and azimuth around the x and z axis
    kite_pos = R_t_w * [0, 0, distance]
    # kite_vel from elevation_vel and azimuth_vel
    kite_vel = R_t_w * [-elevation_vel, azimuth_vel, 0]
    # find quaternion orientation from heading, R_cad_body and R_t_w
    x = [cos(heading), -sin(heading), 0]
    y = [sin(heading),  cos(heading), 0]
    z = [0, 0, 1]
    R_b_w = p.s.wing.R_cad_body * R_t_w * [x y z]
    Q_b_w = rotation_matrix_to_quaternion(R_b_w)
    # adjust the turn rates for observed turn rate
    ω_b = R_b_w' * R_t_w * [0, 0, turn_rate]
    # directly set tether length
    # directly set tether vel
    p.set_state(p.integ, [kite_pos, kite_vel, Q_b_w, ω_b, tether_length, tether_vel])
    display(plot(p.s))
    return nothing
end

function ModelPredictiveControl.preparestate!(p::ModelParams, y)
    set_y!(p, y)
    OrdinaryDiffEq.reinit!(p.integ, p.integ.u; reinit_dae=false)
    stabilize!(p.s, 1.0)
    return p.get_x(p.integ)
end

function ModelPredictiveControl.updatestate!(p::ModelParams, u, y)
    # set_y!(p, y)
    # p.set_u(p.integ, u)
    # OrdinaryDiffEq.reinit!(p.integ, p.integ.u; reinit_dae=false)
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
    # plot_kite(p.s, iter; zoom=true)
    @assert successful_retcode(p.integ.sol)
    xnext = p.get_x(p.integ)
    return xnext
end

function h(x, _, p)
    p.set_x(p.integ, x)
    y = p.get_y(p.integ)
    # y[p.y_idxs[sys.turn_rate[1]]] = 0.0
    # y[p.y_idxs[sys.turn_rate[2]]] = 0.0
    # y[p.y_idxs[sys.turn_acc[1]]] = 0.0
    # y[p.y_idxs[sys.turn_acc[2]]] = 0.0
    # y[p.y_idxs[sys.distance_vel]] = 0.0
    # y[p.y_idxs[sys.distance_acc]] = 0.0
    return y
end

p_plant = ModelParams(s_plant, lin_outputs)
p_model = ModelParams(s_model, lin_outputs, p_plant.x0, p_plant.y0, p_plant.u0)

nu, nx, ny = length(p_plant.u_vec), length(p_plant.x_vec), length(p_plant.y_vec)
vx, vu, vy = string.(p_plant.x_vec), string.(p_plant.u_vec), string.(p_plant.y_vec)
plant = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)

# TODO: linearize on a different core https://www.perplexity.ai/search/using-a-julia-scheduler-run-tw-oKloXmWmSR6YWb47nW_1Gg#0
function calc_dsys(s, u, y)
    vsm_t = @elapsed begin
        s.set_nonstiff(s.integrator, y)
        OrdinaryDiffEq.reinit!(s.integrator, s.integrator.u; reinit_dae=false)
        s.integrator.ps[s.sys.fix_nonstiff] = true
        next_step!(s, u)
        s.integrator.ps[s.sys.fix_nonstiff] = false
    end

    # either use y to find full nonstiff state
    # or have y contain the full nonstiff state

    lin_t = @elapsed begin
        (; A, B, C, D) = KiteModels.linearize(s)
        csys = ss(A, B, C, D)
        if trunc
            tsys, hs, _ = baltrunc_unstab(csys; residual=true, n=length(unknowns))
            # tsys.D .= 0.0
        else
            tsys = csys
        end
        dsys = c2d(tsys, dt, :zoh)
        # dsys = minreal(dsys)
    end
    # @assert norm(dsys.D) ≈ 0
    return dsys, vsm_t, lin_t
end

i_nonstiff = [p_model.y_idxs[sym] for sym in unknowns]
i_ym = [p_model.y_idxs[sym] for sym in measured_states]
dsys, _, _ = calc_dsys(s_model, p_model.u0, p_model.y0[i_nonstiff])
linmodel = ModelPredictiveControl.LinModel(dsys.A, dsys.B, dsys.C, dsys.B[:,end:end-1], dsys.D[:,end:end-1], dt)
setname!(linmodel; u=vu, y=vy)
setop!(linmodel, uop=p_model.u0, yop=p_model.y0)
display(linmodel.A); display(linmodel.Bu)

Hp, Hc, Mwt = 50, 2, fill(0.0, linmodel.ny)
Nwt = [0.01, 0.1, 0.1]
Mwt[p_model.y_idxs[sys.tether_length[1]]] = 10.0
Mwt[p_model.y_idxs[sys.tether_length[2]]] = 1.0
Mwt[p_model.y_idxs[sys.tether_length[3]]] = 1.0
Mwt[p_model.y_idxs[sys.kite_pos[2]]] = 1.0

# Parameter	Lower Value Means	        Higher Value Means
# R	        Less noisy measurements	    More noisy measurements
# Q	        Less noisy model	        More noisy model
# σR = fill(0.1, length(i_ym))
# σQ = fill(0.01, linmodel.nx)
# σQint_u = fill(0.1, linmodel.nu)
# nint_u = fill(1, linmodel.nu)
umin, umax = [-100, -20, -20], [0, 0, 0]
man = ManualEstimator(linmodel; i_ym, nint_u=0, nint_ym=0)
mpc = LinMPC(man; Hp, Hc, Mwt, Nwt, Cwt=Inf)
mpc = setconstraint!(mpc; umin, umax)

KiteModels.linearize_vsm!(p_model.s)
KiteModels.linearize_vsm!(p_plant.s)

function sim_adapt!(mpc, linmodel, N, ry, plant, x0, px0, x̂0, y_step=zeros(plant.ny))
    U_data, Y_data, Ry_data, X̂_data, Ŷ_data = 
        zeros(linmodel.nu, N), zeros(linmodel.ny, N), zeros(linmodel.ny, N), zeros(linmodel.nx, N), zeros(linmodel.ny, N)
    setstate!(plant, px0)
    initstate!(mpc, p_model.u0, plant()[i_ym])
    println("\nStarting simulation...")
    println("─"^80)
    println(
        rpad("Step", 6),
        rpad("Real-time", 12),
        rpad("VSM", 12),
        rpad("Lin", 12),
        rpad("MPC", 12),
        rpad("Estim", 12),
        "Norm(A)"
    )
    println("─"^80)
    for i = 1:N
        @show i
        t = @elapsed begin
            y = plant() + y_step
            estim_t = @elapsed x̂ = preparestate!(p_model, y[i_ym])
            setstate!(mpc, x̂)
            mpc_t = @elapsed u = moveinput!(mpc, ry)

            vsm_t, lin_t = Inf, Inf
            # dsys, vsm_t, lin_t = calc_dsys(s_model, u, y[i_nonstiff])
            # linmodel.A .= dsys.A
            # linmodel.Bu .= dsys.B
            # linmodel.C .= dsys.C
            # setop!(linmodel, uop=p_model.u0, yop=p_model.y0)
            # setmodel!(mpc, linmodel)

            U_data[:,i], Y_data[:,i], Ry_data[:,i], X̂_data[:,i], Ŷ_data[:,i] = u, y, ry, x̂[1:length(x0)], mpc.ŷ
            estim_t += @elapsed x̂ = updatestate!(p_model, u, y[i_ym])       # update UKF state estimate
            setstate!(mpc, x̂)                   # update MPC with the UKF updated state
        end

        KiteModels.linearize_vsm!(p_plant.s)
        updatestate!(plant, u)

        @printf("%4d │ %8.3fx │ %8.3fx │ %8.3fx │ %8.1fx │ %8.1fx │ %.2e\n",
            i, dt/t, dt/vsm_t, dt/lin_t, dt/mpc_t, dt/estim_t, norm(linmodel.A))
    end
    res = SimResult(mpc, U_data, Y_data; Ry_data, X̂_data, Ŷ_data)
    return res
end

ry = p_model.y0
# ry[p_model.y_idxs[sys.kite_pos[2]]] = 1.0
# ry[p_model.y_idxs[sys.angle_of_attack]] = deg2rad(10)

x0 = p_model.get_x(p_model.integ)
x̂0 = x0
# x̂0 = zeros(estim.nx̂)
res = sim_adapt!(mpc, linmodel, N, ry, plant, x0, p_plant.x0, x̂0)
y_idxs = [
    findall(x -> x != 0.0, Mwt)
    # [p_model.y_idxs[sys.Q_b_w[i]] for i in 1:4]
    # p_model.y_idxs[sys.azimuth]
    p_model.y_idxs[sys.heading]
    # p_model.y_idxs[sys.kite_pos[3]]
]
Plots.plot(res; plotx=false, ploty=y_idxs, 
    plotxwithx̂=[141],
    plotŷ=true,
    plotu=true, size=(900, 900)
)
