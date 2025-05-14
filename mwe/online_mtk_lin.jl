"""
Use a nonlinear MTK model with online linearization and a linear MPC using ModelPredictiveControl.jl
"""

using KiteModels, KiteUtils, LinearAlgebra
using ModelPredictiveControl, ModelingToolkit, Plots, JuMP, Ipopt, OrdinaryDiffEq, DifferentiationInterface,
    SymbolicIndexingInterface
using ModelingToolkit: setu, setp, getu, getp, @variables
using ModelingToolkit: t_nounits as t
using SciMLBase: successful_retcode
using RobustAndOptimalControl
using ControlSystems
using Plots
using Printf

include("utils.jl")
set_data_path(joinpath(dirname(@__DIR__), "data"))

ad_type = AutoForwardDiff()
N = 200
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

measured_states = [
    sys.elevation
    sys.elevation_vel
    sys.azimuth
    sys.azimuth_vel
    sys.heading_x # TODO: maybe can be removed, should at least be relative to straight up
    sys.turn_rate
    sys.tether_length...
    sys.tether_vel...
]
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
stabilize!(s_model)
stabilize!(s_plant)

# https://github.com/SciML/ModelingToolkit.jl/issues/3552#issuecomment-2817642041
function f(x, u, _, p)
    p.set_x(p.integ, x)
    p.set_u(p.integ, u)
    OrdinaryDiffEq.reinit!(p.integ, p.integ.u; reinit_dae=false)
    OrdinaryDiffEq.step!(p.integ, dt)
    # plot_kite(p.s, iter; zoom=true)
    if !successful_retcode(p.integ.sol)
        @show x u p.sx
        @assert successful_retcode(p.integ.sol)
    end
    xnext = p.get_x(p.integ)
    !successful_retcode(p.integ.sol) && (xnext .= NaN)
    return xnext
end

function h(x, _, p)
    p.set_x(p.integ, x)
    y = p.get_y(p.integ)
    return y
end

p_model = ModelParams(s_model, lin_outputs)
p_plant = ModelParams(s_plant, lin_outputs)

nu, nx, ny = length(p_model.u_vec), length(p_model.x_vec), length(p_model.y_vec)
vx, vu, vy = string.(p_model.x_vec), string.(p_model.u_vec), string.(p_model.y_vec)
model = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_model, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
setstate!(model, p_model.x0)

nu, nx, ny = length(p_plant.u_vec), length(p_plant.x_vec), length(p_plant.y_vec)
vx, vu, vy = string.(p_plant.x_vec), string.(p_plant.u_vec), string.(p_plant.y_vec)
plant = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)

Hp, Hc, Mwt = 50, 2, fill(0.0, model.ny)
Nwt = [0.01, 0.1, 0.1]
Mwt[p_model.y_idxs[sys.tether_length[1]]] = 10.0
Mwt[p_model.y_idxs[sys.tether_length[2]]] = 1.0
Mwt[p_model.y_idxs[sys.tether_length[3]]] = 1.0
Mwt[p_model.y_idxs[sys.kite_pos[2]]] = 1.0

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
        dsys = c2d(tsys, dt)
    end
    # @assert norm(dsys.D) ≈ 0
    return dsys, vsm_t, lin_t
end

nonstiff_idxs = [p_model.y_idxs[sym] for sym in unknowns]
measured_idxs = [p_model.y_idxs[sym] for sym in measured_states]
dsys, _, _ = calc_dsys(s_model, p_model.u0, p_model.y0[nonstiff_idxs])
linmodel = ModelPredictiveControl.LinModel(dsys.A, dsys.B, dsys.C, dsys.B[:,end:end-1], dsys.D[:,end:end-1], dt)
setname!(linmodel; u=vu, y=vy)
setop!(linmodel, uop=p_model.u0, yop=p_model.y0)
display(linmodel.A); display(linmodel.Bu)

# Parameter	Lower Value Means	        Higher Value Means
# R	        Less noisy measurements	    More noisy measurements
# Q	        Less noisy model	        More noisy model
σR = zeros(linmodel.ny)
σR[nonstiff_idxs] .= 1e12
σR[measured_idxs] .= 0.1
σQ = fill(1.0, linmodel.nx)
σQint_u = fill(0.1, linmodel.nu)
nint_u = fill(1, linmodel.nu)
umin, umax = [-100, -20, -20], [0, 0, 0]
estim = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
mpc = setconstraint!(mpc; umin, umax)

# plot_lin_precision()
KiteModels.linearize_vsm!(p_model.s)
KiteModels.linearize_vsm!(p_plant.s)

function sim_adapt!(mpc, nonlinmodel, N, ry, plant, x0, px0, x̂0, y_step=zeros(ny))
    U_data, Y_data, Ry_data, X̂_data, Ŷ_data = 
        zeros(linmodel.nu, N), zeros(linmodel.ny, N), zeros(linmodel.ny, N), zeros(linmodel.nx, N), zeros(linmodel.ny, N)
    setstate!(plant, px0)
    initstate!(mpc, p_model.u0, plant())
    setstate!(mpc, x̂0)
    println("\nStarting simulation...")
    println("─"^80)
    println(
        rpad("Step", 6),
        rpad("Real-time", 12),
        rpad("VSM", 12),
        rpad("Lin", 12),
        rpad("MPC", 12),
        "Norm(A)"
    )
    println("─"^80)
    for i = 1:N
        t = @elapsed begin
            y = plant() + y_step
            x̂ = preparestate!(mpc, y)[1:length(x0)]
            u = moveinput!(mpc, ry)

            dsys, vsm_t, lin_t = calc_dsys(s_model, u, y[nonstiff_idxs])
            linmodel.A .= dsys.A
            linmodel.Bu .= dsys.B
            linmodel.C .= dsys.C
            setop!(linmodel, uop=p_model.u0, yop=p_model.y0)
            setmodel!(mpc, linmodel)

            U_data[:,i], Y_data[:,i], Ry_data[:,i], X̂_data[:,i], Ŷ_data[:,i] = u, y, ry, x̂, mpc.ŷ
            mpc_t = @elapsed updatestate!(mpc, u, y) # update mpc state estimate
        end

        KiteModels.linearize_vsm!(p_plant.s)
        updatestate!(plant, u)
        # plot_kite(s_plant, i-1; zoom=true)

        @printf("%4d │ %8.3fx │ %8.3fx │ %8.3fx │ %8.1fx │ %.2e\n",
            i, dt/t, dt/vsm_t, dt/lin_t, dt/mpc_t, norm(linmodel.A))
    end
    res = SimResult(mpc, U_data, Y_data; Ry_data, X̂_data, Ŷ_data)
    return res
end

ry = p_model.y0
# ry[p_model.y_idxs[sys.kite_pos[2]]] = 1.0
# ry[p_model.y_idxs[sys.angle_of_attack]] = deg2rad(10)

# M = [
#     (I - linmodel.A);
#     linmodel.C
# ]
# b = [
#     linmodel.Bu * linmodel.uop;
#     linmodel.yop
# ]
x0 = zeros(linmodel.nx)
x̂0 = [
    x0
    p_model.y0
]
res = sim_adapt!(mpc, model, N, ry, plant, x0, p_plant.x0, x̂0)
y_idxs = [
    findall(x -> x != 0.0, Mwt)
    [p_model.y_idxs[sys.Q_b_w[i]] for i in 1:4]
]
Plots.plot(res; plotx=false, ploty=y_idxs, 
    plotxwithx̂=false,
    plotŷ=true,
    plotu=true, size=(900, 900)
)
