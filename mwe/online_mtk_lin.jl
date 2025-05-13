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

set_data_path(joinpath(dirname(@__DIR__), "data"))

ad_type = AutoForwardDiff()
N = 100
trunc = true
u_idxs = [1,2,3]

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
    sys.elevation_acc
    sys.azimuth
    sys.azimuth_vel
    sys.azimuth_acc
    sys.heading_x
    sys.tether_length...
    sys.tether_vel...
]
KiteModels.init_sim!(s_model, measure; remake=false, adaptive=true, lin_outputs=measured_states)

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

# Get initial state
struct ModelParams{G1,G2}
    s::RamAirKite
    integ::OrdinaryDiffEq.ODEIntegrator
    set_x::Union{SymbolicIndexingInterface.MultipleSetters, Nothing}
    set_ix::Union{SymbolicIndexingInterface.MultipleSetters, Nothing}
    set_u::SymbolicIndexingInterface.MultipleSetters
    get_x::G1
    get_y::G2
    dt::Float64
    
    x_vec::Vector{Num}
    y_vec::Vector{Num}
    u_vec::Vector{Num}
    x0::Vector{Float64}
    y0::Vector{Float64}
    u0::Vector{Float64}
    x_idxs::Dict{Num, Int64}
    y_idxs::Dict{Num, Int64}
    Q_idxs::Vector{Int64}
    function ModelParams(s, y_vec)
        # [println(x) for x in x_vec]
        # y_vec = [
        #     [s.sys.tether_length[i] for i in 1:3]
        #     s.sys.elevation
        #     s.sys.azimuth
        # ]
        x_vec = KiteModels.get_unknowns(s_model)
        u_vec = [s.sys.set_values[u_idxs[i]] for i in eachindex(u_idxs)]

        set_x = setu(s.integrator, x_vec)
        set_ix = setu(s.integrator, Initial.(x_vec))
        set_u = setu(s.integrator, u_vec)
        get_x = getu(s.integrator, x_vec)
        get_y = getu(s.integrator, y_vec)
        x0 = get_x(s.integrator)
        y0 = get_y(s.integrator)
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
        
        return new{typeof(get_x), typeof(get_y)}(
            s, s.integrator, set_x, set_ix, set_u, 
            get_x, get_y, dt, x_vec, y_vec, u_vec,
            x0, y0, u0, x_idxs, y_idxs, Q_idxs
        )
    end
end

p_model = ModelParams(s_model, measured_states)
p_plant = ModelParams(s_plant, measured_states)

nu, nx, ny = length(p_model.u_vec), length(p_model.x_vec), length(p_model.y_vec)
vx, vu, vy = string.(p_model.x_vec), string.(p_model.u_vec), string.(p_model.y_vec)
model = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_model, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
setstate!(model, p_model.x0)

nu, nx, ny = length(p_plant.u_vec), length(p_plant.x_vec), length(p_plant.y_vec)
vx, vu, vy = string.(p_plant.x_vec), string.(p_plant.u_vec), string.(p_plant.y_vec)
plant = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)

Hp, Hc, Mwt, Nwt = 5, 2, fill(0.0, model.ny), fill(0.01, model.nu)
Mwt[p_model.y_idxs[sys.tether_length[1]]] = 10.0
Mwt[p_model.y_idxs[sys.tether_length[2]]] = 1.0
Mwt[p_model.y_idxs[sys.tether_length[3]]] = 1.0
# Mwt[p_model.y_idxs[sys.angle_of_attack]] = rad2deg(1.0)
# Mwt[p_model.y_idxs[sys.kite_pos[2]]] = 1.0
# Mwt[p_model.y_idxs[sys.heading_x]] = 1.0

# TODO: linearize on a different core https://www.perplexity.ai/search/using-a-julia-scheduler-run-tw-oKloXmWmSR6YWb47nW_1Gg#0
function calc_tsys(s)
    (; A, B, C, D) = KiteModels.linearize(s)
    @show norm(A)
    csys = ss(A, B, C, D)
    if trunc
        tsys, hs, _ = baltrunc_unstab(csys; residual=true, n=24)
        @show norm(tsys.D)
        tsys.D .= 0.0
    else
        tsys = csys
    end
    return tsys
end

tsys = calc_tsys(s_model)
@time linmodel = ModelPredictiveControl.LinModel(tsys, dt)
setname!(linmodel; u=vu, y=vy)
setop!(linmodel, uop=p_model.u0, yop=p_model.y0)
display(linmodel.A); display(linmodel.Bu)

# Parameter	Lower Value Means	        Higher Value Means
# R	        More trust in measurements	Less trust in measurements
# Q	        More trust in model	        Less trust in model
σR = fill(0.1, linmodel.ny)
σQ = fill(0.1, linmodel.nx)
σQint_u = fill(0.1, linmodel.nu)
nint_u = fill(1, linmodel.nu)
umin, umax = [-100, -20, -20][u_idxs], [0, 0, 0][u_idxs]
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
            
            U_data[:,i], Y_data[:,i], Ry_data[:,i], X̂_data[:,i], Ŷ_data[:,i] = u, y, ry, x̂, mpc.ŷ
            mpc_t = @elapsed updatestate!(mpc, u, y) # update mpc state estimate
        end

        KiteModels.linearize_vsm!(p_plant.s)
        updatestate!(plant, u)
        # plot_kite(s_plant, i-1; zoom=true)

        vsm_t, lin_t = zeros(2)
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
y_idxs = findall(x -> x != 0.0, Mwt)
Plots.plot(res; plotx=false, ploty=y_idxs, 
    plotxwithx̂=false,
    plotŷ=true,
    plotu=true, size=(900, 900)
)
