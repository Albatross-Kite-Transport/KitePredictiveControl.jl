# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

"""
Use a nonlinear MTK model with online linearization and a linear MPC using ModelPredictiveControl.jl
"""

using KiteModels, KiteUtils, LinearAlgebra
using ModelPredictiveControl, ModelingToolkit, Plots, JuMP, Ipopt, OrdinaryDiffEq, FiniteDiff, DifferentiationInterface,
    SymbolicIndexingInterface
using ModelingToolkit: setu, setp, getu, getp
using SciMLBase: successful_retcode
using ControlPlots
using Printf

set_data_path(joinpath(dirname(@__DIR__), "data"))
include("plotting.jl")
include("plot_lin_precision.jl")

ad_type = AutoFiniteDiff()
N = 20
u_idxs = [1,2,3]

# Initialize model
set_model = deepcopy(load_settings("system_model.yaml"))
set_plant = deepcopy(load_settings("system_plant.yaml"))
s_model = SymbolicAWEModel(set_model)
s_plant = SymbolicAWEModel(set_plant)
dt = 1/set_model.sample_freq

measure = Measurement()
measure.sphere_pos .= deg2rad.([70.0 70.0; 1.0 -1.0])
KiteModels.init_sim!(s_model, measure; 
    adaptive=true, remake=false, reload=true, 
    solver=FBDF(nlsolve=OrdinaryDiffEq.NLNewton(relax=0.8, max_iter=1000))
)
OrdinaryDiffEq.set_proposed_dt!(s_model.integrator, 0.1dt)
KiteModels.init_sim!(s_plant, measure; remake=false, adaptive=true)
sys = s_model.sys

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
# OrdinaryDiffEq.set_proposed_dt!(s_model.integrator, 0.01dt)

"""
Use quick initialization model to extend the measurements for the kalman filter
"""

# https://github.com/SciML/ModelingToolkit.jl/issues/3552#issuecomment-2817642041
function f(x, u, _, p)
    p.set_x(p.integ, x)
    p.set_sx(p.integ, p.sx)
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

# Get initial state
struct ModelParams{G1,G2,G3}
    s::SymbolicAWEModel
    integ::OrdinaryDiffEq.ODEIntegrator
    set_x::Union{SymbolicIndexingInterface.MultipleSetters, Nothing}
    set_ix::Union{SymbolicIndexingInterface.MultipleSetters, Nothing}
    set_sx::Union{SymbolicIndexingInterface.MultipleSetters, Nothing}
    sx::Vector{Float64}
    set_u::SymbolicIndexingInterface.MultipleSetters
    get_x::G1
    get_sx::G2
    get_y::G3
    dt::Float64
    
    x_vec::Vector{Num}
    sx_vec::Vector{Num}
    y_vec::Vector{Num}
    u_vec::Vector{Num}
    sx0::Vector{Float64}
    x0::Vector{Float64}
    u0::Vector{Float64}
    x_idxs::Dict{Num, Int64}
    y_idxs::Dict{Num, Int64}
    Q_idxs::Vector{Int64}
    function ModelParams(s, x_vec, sx_vec)
        # [println(x) for x in x_vec]
        # y_vec = [
        #     [s.sys.tether_length[i] for i in 1:3]
        #     s.sys.elevation
        #     s.sys.azimuth
        # ]
        y_vec = [x_vec; s.sys.angle_of_attack]
        u_vec = [s.sys.set_values[u_idxs[i]] for i in eachindex(u_idxs)]

        set_x = setu(s.integrator, x_vec)
        set_ix = setu(s.integrator, Initial.(x_vec))
        set_sx = setu(s.integrator, sx_vec)
        set_u = setu(s.integrator, u_vec)
        get_x = getu(s.integrator, x_vec)
        get_sx = getu(s.integrator, sx_vec)
        get_y = getu(s.integrator, y_vec)
        sx = get_sx(s.integrator)
        sx0 = copy(sx)
        x0 = get_x(s.integrator)
        u0 = -s.set.drum_radius * s.integrator[s.sys.winch_force][u_idxs]
        u0 = [-50.0, -1.0, -1.0][u_idxs]
        x_idxs = Dict{Num, Int}()
        for (idx, sym) in enumerate(x_vec)
            x_idxs[sym] = idx
        end
        y_idxs = Dict{Num, Int}()
        for (idx, sym) in enumerate(y_vec)
            y_idxs[sym] = idx
        end
        Q_idxs = [x_idxs[s.sys.Q_b_w[i]] for i in 1:4]
        
        return new{typeof(get_x), typeof(get_sx), typeof(get_y)}(
            s, s.integrator, set_x, set_ix, set_sx, sx, set_u, 
            get_x, get_sx, get_y, dt, x_vec, sx_vec, y_vec, u_vec,
            sx0, x0, u0, x_idxs, y_idxs, Q_idxs
        )
    end
end

x_vec = KiteModels.get_unknowns(s_model)
p_model = ModelParams(s_model, x_vec, Num[])
sx_vec = setdiff(KiteModels.get_unknowns(s_plant), x_vec)
p_plant = ModelParams(s_plant, x_vec, sx_vec)

test_model(p_model)

nu, nx, ny = length(p_model.u_vec), length(p_model.x_vec), length(p_model.y_vec)
vx, vu, vy = string.(p_model.x_vec), string.(p_model.u_vec), string.(p_model.y_vec)
model = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_model, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
setstate!(model, p_model.x0)

nu, nx, ny = length(p_plant.u_vec), length(p_plant.x_vec), length(p_plant.y_vec)
vx, vu, vy = string.(p_plant.x_vec), string.(p_plant.u_vec), string.(p_plant.y_vec)
plant = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)

Hp, Hc, Mwt, Nwt = 20, 2, fill(0.0, model.ny), fill(1.0, model.nu)
Mwt[p_model.y_idxs[sys.tether_length[1]]] = 1.0
Mwt[p_model.y_idxs[sys.angle_of_attack]] = rad2deg(1.0)
Mwt[p_model.y_idxs[sys.kite_pos[2]]] = 1.0


# TODO: linearize on a different core https://www.perplexity.ai/search/using-a-julia-scheduler-run-tw-oKloXmWmSR6YWb47nW_1Gg#0
prepare_model!(p_model, p_model.x0)
@time linmodel = ModelPredictiveControl.linearize(model, x=p_model.x0, u=p_model.u0)
display(linmodel.A); display(linmodel.Bu)

# Parameter	Lower Value Means	        Higher Value Means
# R	        More trust in measurements	Less trust in measurements
# Q	        More trust in model	        Less trust in model
σR = fill(0.1, model.ny)
σQ = fill(1.0, model.nx)
σQint_u = fill(0.1, model.nu)
nint_u = fill(1, model.nu)
umin, umax = [-100, -20, -20][u_idxs], [0, 0, 0][u_idxs]
estim = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
mpc = setconstraint!(mpc; umin, umax)

# plot_lin_precision()
reset_p!(p_model)
reset_p!(p_plant)
KiteModels.linearize_vsm!(p_model.s)
KiteModels.linearize_vsm!(p_plant.s)

plot_lin_precision()

# function sim_adapt!(mpc, nonlinmodel, N, ry, plant, x0, px0, x̂0, y_step=zeros(ny))
#     U_data, Y_data, Ry_data, X̂_data, X_data = 
#         zeros(model.nu, N), zeros(model.ny, N), zeros(model.ny, N), zeros(model.nx, N), zeros(model.nx, N)
#     setstate!(plant, px0)
#     initstate!(mpc, p_model.u0, plant())
#     setstate!(mpc, x̂0)
#     println("\nStarting simulation...")
#     println("─"^80)
#     println(
#         rpad("Step", 6),
#         rpad("Real-time", 12),
#         rpad("VSM", 12),
#         rpad("Lin", 12),
#         rpad("MPC", 12),
#         rpad("Norm(A)", 12),
#         "success"
#     )
#     println("─"^80)
#     success = false
#     for i = 1:N
#         t = @elapsed begin
#             y = plant() + y_step
#             x̂ = preparestate!(mpc, y)[1:length(x0)]
#             u = moveinput!(mpc, ry)
            
#             vsm_t = @elapsed KiteModels.linearize_vsm!(p_model.s)
#             prepare_model!(p_model, x̂)
            
#             # smooth = 0.9
#             # lin_t = @elapsed next_linmodel = ModelPredictiveControl.linearize(nonlinmodel; u, x=x̂)
#             # @. linmodel.A = (1-smooth)*next_linmodel.A + smooth*linmodel.A
#             # @. linmodel.Bu = (1-smooth)*next_linmodel.Bu + smooth*linmodel.Bu
#             # @. linmodel.C = (1-smooth)*next_linmodel.C + smooth*linmodel.C
#             lin_t = @elapsed ModelPredictiveControl.linearize!(linmodel, nonlinmodel; u, x=x̂)

#             success = false
#             if !(any(isnan.(linmodel.A)) || any(isnan.(linmodel.Bu)))
#                 success = true
#                 setmodel!(mpc, linmodel)
#             end
            
#             U_data[:,i], Y_data[:,i], Ry_data[:,i], X̂_data[:,i] = u, y, ry, x̂
#             mpc_t = @elapsed updatestate!(mpc, u, y) # update mpc state estimate
#         end

#         p_plant.sx .= p_plant.get_sx(p_plant.integ)
#         KiteModels.linearize_vsm!(p_plant.s)
#         updatestate!(plant, u)
#         plot_kite(s_plant, i-1; zoom=true)

#         X_data[:,i] .= p_plant.get_y(p_plant.integ)[1:length(x0)]
#         @printf("%4d │ %8.3fx │ %8.3fx │ %8.3fx │ %8.1fx │ %.2e | %d\n",
#             i, dt/t, dt/vsm_t, dt/lin_t, dt/mpc_t, norm(linmodel.A), success)
#     end
#     res = SimResult(mpc, U_data, Y_data; Ry_data, X̂_data, X_data)
#     return res
# end

# ry = p_model.get_y(s_model.integrator)
# ry[p_model.y_idxs[sys.kite_pos[2]]] = 1.0
# # ry[p_model.y_idxs[sys.angle_of_attack]] = deg2rad(10)

# x̂0 = [
#     p_model.x0
#     p_model.get_y(s_model.integrator)
# ]
# res = sim_adapt!(mpc, model, N, ry, plant, p_model.x0, p_plant.x0, x̂0)
# y_idxs = findall(x -> x != 0.0, Mwt)
# Plots.plot(res; plotx=false, ploty=y_idxs, 
#     plotxwithx̂=[
#         [p_model.x_idxs[sys.tether_length[i]] for i in 1:3]
#         [p_model.x_idxs[sys.Q_b_w[i]] for i in 1:4]
#     ], 
#     plotu=true, size=(900, 900)
# )
