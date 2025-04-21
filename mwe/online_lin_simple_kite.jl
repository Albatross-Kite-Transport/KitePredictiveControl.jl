"""
Use a nonlinear MTK model with online linearization and a linear MPC using ModelPredictiveControl.jl
"""

using KiteModels, KiteUtils, LinearAlgebra
using ModelPredictiveControl, ModelingToolkit, Plots, JuMP, Ipopt, OrdinaryDiffEq, FiniteDiff, DifferentiationInterface,
    SymbolicIndexingInterface
using ModelingToolkit: setu, setp, getu, getp
using SciMLBase: successful_retcode
using ControlPlots

set_data_path(joinpath(dirname(@__DIR__), "data"))
include("plotting.jl")
include("plot_lin_precision.jl")

ad_type = AutoFiniteDiff(absstep=1e-5, relstep=1e-5)
u_idxs = [1]

# Initialize model
set_model = deepcopy(load_settings("system_model.yaml"))
set_plant = deepcopy(load_settings("system_plant.yaml"))
s_model = RamAirKite(set_model)
s_plant = RamAirKite(set_plant)
dt = 1/set_model.sample_freq

measure = Measurement()
measure.sphere_pos .= deg2rad.([83.0 83.0; 1.0 -1.0])
KiteModels.init_sim!(s_model, measure; remake=false, adaptive=false)
KiteModels.init_sim!(s_plant, measure; remake=false, adaptive=false)
OrdinaryDiffEq.set_proposed_dt!(s_model.integrator, 0.01dt)
OrdinaryDiffEq.set_proposed_dt!(s_plant.integrator, 0.01dt)
sys = s_model.sys

function stabilize!(s)
    s.integrator.ps[sys.steady] = true
    next_step!(s; dt=10.0, vsm_interval=1)
    s.integrator.ps[sys.steady] = false
end
stabilize!(s_model)
stabilize!(s_plant)

"""
current problem: initialization works only at x0
solution: use dynamic model
- initialize with acc = 0 for tether points at the start of each time step
- reinit with reinit_dae=false inside discrete function

The simplest way to get the stiff unknowns:
- step model with the found inputs u to find the next step stiff unknowns
- might be problematic if the model is way off the real system
"""

function f(x, u, _, p)
    p.set_x(p.integ, x)
    p.set_sx(p.integ, p.sx)
    p.set_u(p.integ, u)
    OrdinaryDiffEq.set_t!(p.integ, 0.0)
    OrdinaryDiffEq.step!(p.integ, p.dt)
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
struct ModelParams
    s::RamAirKite
    integ::OrdinaryDiffEq.ODEIntegrator
    set_x::SymbolicIndexingInterface.MultipleSetters
    set_ix::SymbolicIndexingInterface.MultipleSetters
    set_sx::SymbolicIndexingInterface.MultipleSetters
    sx::Vector{Float64}
    set_u::SymbolicIndexingInterface.MultipleSetters
    get_x::SymbolicIndexingInterface.MultipleGetters
    get_sx::SymbolicIndexingInterface.MultipleGetters
    get_y::SymbolicIndexingInterface.MultipleGetters
    dt::Float64

    x_vec::Vector{Num}
    sx_vec::Vector{Num}
    y_vec::Vector{Num}
    u_vec::Vector{Num}
    function ModelParams(s)
        x_vec = KiteModels.get_nonstiff_unknowns(s)
        sx_vec = KiteModels.get_stiff_unknowns(s)
        y_vec = [s.sys.tether_length[1]]
        u_vec = [s.sys.set_values[u_idxs[i]] for i in eachindex(u_idxs)]

        set_x = setu(s.integrator, x_vec)
        set_ix = setu(s.integrator, Initial.(x_vec))
        set_sx = setu(s.integrator, sx_vec)
        set_u = setu(s.integrator, u_vec)
        get_x = getu(s.integrator, x_vec)
        get_sx = getu(s.integrator, sx_vec)
        get_y = getu(s.integrator, y_vec)
        sx = get_sx(s.integrator)
        return new(s, s.integrator, set_x, set_ix, set_sx, sx, set_u, get_x, get_sx, get_y, dt, x_vec, sx_vec, y_vec, u_vec)
    end
end

p_model = ModelParams(s_model)
p_plant = ModelParams(s_plant)
nu, nx, ny = length(p_model.u_vec), length(p_model.x_vec), length(p_model.y_vec)

x0 = p_model.get_x(s_model.integrator)
sx0 = p_model.get_sx(s_model.integrator)
u0 = [-s_model.set.drum_radius * s_model.integrator[sys.winch_force][1]] .+ 10

norms = Float64[]
for x in [x0, x0 .+ 0.01]
    for u in [[-50, 0, 0], [-51, -1, -1]]
        for _ in 1:2
            xnext = f(x, u, nothing, p_model)
            push!(norms, norm(xnext))
            ynext = h(xnext, nothing, p_model)
            # @info "x: $(norm(x)) u: $(norm(u)) xnext: $(norm(xnext)) ynext: $(norm(ynext))"
        end
    end
end
@assert length(unique(norms))*2 == length(norms) "Different inputs/states should produce different outputs"

x_idx = Dict{Num, Int}()
for (idx, sym) in enumerate(p_model.x_vec)
    x_idx[sym] = idx
end
y_idx = Dict{Num, Int}()
for (idx, sym) in enumerate(p_model.y_vec)
    y_idx[sym] = idx
end

vx = string.(p_model.x_vec)
vu = string.(p_model.u_vec)
vy = string.(p_model.y_vec)
model = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_model, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
setstate!(model, x0)
# setop!(model; xop=x0)

u = [-50, -5, 0][u_idxs]
N = 10
# res = sim!(model, N, u; x_0=x0)
# display(plot(res; plotx=false, ploty=[11,12,13,14,15,16], plotu=false, size=(900, 900)))

plant = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)

Hp, Hc, Mwt, Nwt = 3, 1, fill(1.0, ny), fill(0.1, nu)
# Mwt[y_idx[sys.tether_length[1]]] = 0.01
# Mwt[y_idx[sys.tether_length[2]]] = 1.0
# Mwt[y_idx[sys.tether_length[3]]] = 1.0

# TODO: linearize on a different core https://www.perplexity.ai/search/using-a-julia-scheduler-run-tw-oKloXmWmSR6YWb47nW_1Gg#0
@time linmodel = ModelPredictiveControl.linearize(model, x=x0, u=u0)
display(linmodel.A); display(linmodel.Bu)

σR = fill(0.001, ny)
σQ = fill(0.001, nx)
σQint_u = fill(0.1, nu)
nint_u = fill(1, nu)
umin, umax = [-100, -20, -20][u_idxs], [0, 0, 0][u_idxs]
estim = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
mpc = setconstraint!(mpc; umin, umax)

# plot_lin_precision()

function sim_adapt!(mpc, nonlinmodel, N, ry, plant, x0, x̂0, y_step=zeros(ny))
    U_data, Y_data, Ry_data, X̂_data, X_data = 
        zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N), zeros(plant.nx, N), zeros(plant.nx, N)
    setstate!(plant, x0)
    initstate!(mpc, u0, plant())
    setstate!(mpc, x̂0)
    for i = 1:N
        t = @elapsed begin
            y = plant() + y_step
            x̂ = preparestate!(mpc, y)
            u = moveinput!(mpc, ry)
            
            sx0, x0 = p_model.get_sx(p_model.integ), x̂[1:length(x0)]
            reset_p!(p_model, sx0, x0)
            reset_p!(p_plant, sx0)
            # setstate!(model, x0)
            # setstate!(plant, x0)        

            linmodel = ModelPredictiveControl.linearize(nonlinmodel; u, x=x̂[1:length(x0)])
            @show norm(linmodel.A)
            setmodel!(mpc, linmodel)
            @show linmodel.Bu[x_idx[sys.tether_vel[1]]]

            U_data[:,i], Y_data[:,i], Ry_data[:,i], X̂_data[:,i] = u, y, ry, x̂[1:length(x0)]
            updatestate!(mpc, u, y) # update mpc state estimate
        end
        plot_kite(s_plant, i-1)
        updatestate!(plant, u)  # update plant simulator
        X_data[:,i] .= plant.x0
        println("$(dt/t) times realtime at timestep $i. Norm A: $(norm(linmodel.A)). Norm Bu: $(norm(linmodel.Bu)). Norm vsm_jac: $(norm(s_model.prob.ps[sys.vsm_jac]))")
    end
    res = SimResult(mpc, U_data, Y_data; Ry_data, X̂_data, X_data)
    return res
end

ry = p_model.get_y(s_model.integrator)
x̂0 = [
    x0
    p_model.get_y(s_model.integrator)
]
res = sim_adapt!(mpc, model, N, ry, plant, x0, x̂0)
y_idxs = findall(x -> x != 0.0, Mwt)
Plots.plot(res; plotx=false, ploty=y_idxs, 
    plotxwithx̂=[
        [x_idx[sys.tether_length[i]] for i in 1:3]
        [x_idx[sys.Q_b_w[i]] for i in 1:4]
    ], 
    plotu=true, size=(900, 900)
)

# plot_lin_precision()
