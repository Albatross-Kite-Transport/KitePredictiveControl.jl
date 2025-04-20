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
include(joinpath(@__DIR__, "plotting.jl"))

ad_type = AutoFiniteDiff(absstep=0.01, relstep=0.01)

# Initialize model
set_model = deepcopy(load_settings("system_model.yaml"))
set_plant = deepcopy(load_settings("system_plant.yaml"))
s_model = RamAirKite(set_model)
s_plant = RamAirKite(set_plant)
dt = 1/set_model.sample_freq

measure = Measurement()
measure.sphere_pos .= deg2rad.([83.0 83.0; 1.0 -1.0])
KiteModels.init_sim!(s_model, measure; remake=false)
KiteModels.init_sim!(s_plant, measure; remake=false)
sys = s_model.sys

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
    OrdinaryDiffEq.reinit!(p.integ, p.integ.u; reinit_dae=false)
    OrdinaryDiffEq.step!(p.integ, p.dt)
    xnext = p.get_x(p.integ)
    !successful_retcode(p.integ.sol) && (xnext .= NaN)
    return xnext
end

function f_plant(x, u, _, p)
    next_step!(p.s, u; p.dt, vsm_interval=0)
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
        y_vec = collect(s.sys.tether_length)
        u_vec = collect(s.sys.set_values)

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

x0 = p_model.get_x(s_model.integrator)
u0 = -s_model.set.drum_radius * s_model.integrator[sys.winch_force]

nu, nx, ny = length(p_model.u_vec), length(p_model.x_vec), length(p_model.y_vec)
norms = Float64[]
for x in [x0, x0 .+ 0.01]
    for u in [[-50, 0, 0], [-50, -1, -1]]
        xnext = f(x, u, nothing, p_model)
        push!(norms, norm(xnext))
        ynext = h(xnext, nothing, p_model)
        @info "x: $(norm(x)) u: $(norm(u)) xnext: $(norm(xnext)) ynext: $(norm(ynext))"
    end
end
@assert length(unique(norms)) == length(norms) "Different inputs/states should produce different outputs"

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

u = [-50, -5, 0]
N = 50
# res = sim!(model, N, u; x_0=x0)
# display(plot(res; plotx=false, ploty=[11,12,13,14,15,16], plotu=false, size=(900, 900)))

α=0.01
σR = fill(0.01, ny)
σQ = fill(0.01, nx)
σQint_u = fill(0.1, nu)
nint_u = fill(1, nu)

plant = setname!(NonLinModel(f_plant, h, dt, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
# estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)
# res = sim!(estim, N, [-50, -0.1, -0.1]; x_0=x0, plant=plant, y_noise=fill(0.01, ny))
# plot(res; plotx=false, ploty=[11,12,13,14,15,16,17], plotu=false, plotxwithx̂=false, size=(900, 900))

Hp, Hc, Mwt, Nwt = 2, 1, fill(1.0, ny), fill(0.1, nu)

# TODO: linearize on a different core https://www.perplexity.ai/search/using-a-julia-scheduler-run-tw-oKloXmWmSR6YWb47nW_1Gg#0
@time linmodel = ModelPredictiveControl.linearize(model, x=x0, u=u0)
display(linmodel.A); display(linmodel.Bu)

umin, umax = [-100, -20, -20], [0, 0, 0]
Δumin, Δumax = [-10,-1,-1], [10,1,1]

estim = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
mpc = setconstraint!(mpc; umin, umax)

include("plot_lin_precision.jl")

# function sim_adapt!(mpc, nonlinmodel, N, ry, plant, x0, x̂_0, y_step=zeros(ny))
#     U_data, Y_data, Ry_data, X̂_data = zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N), zeros(plant.nx, N)
#     setstate!(plant, x0)
#     initstate!(mpc, u0, plant())
#     setstate!(mpc, x̂_0)
#     for i = 1:N
#         t = @elapsed begin
            
#             y = plant() + y_step
#             x̂ = preparestate!(mpc, y)
#             u = moveinput!(mpc, ry)
            
#             vsm_y = s_plant.get_y(s_plant.integrator) # TODO: use model
#             vsm_jac, vsm_x = VortexStepMethod.linearize(
#                 s_model.vsm_solver, 
#                 s_model.aero, 
#                 vsm_y;
#                 va_idxs=1:3, 
#                 omega_idxs=4:6,
#                 theta_idxs=7:6+length(s_model.point_system.groups),
#                 moment_frac=s_model.bridle_fracs[s_model.point_system.groups[1].fixed_index])
#             s_model.set_vsm(s_model.prob, [vsm_x, vsm_y, vsm_jac])
#             s_plant.set_vsm(s_plant.integrator, [vsm_x, vsm_y, vsm_jac])

#             linearize!(linmodel, nonlinmodel; u, x=x̂[1:length(x0)])
#             setmodel!(mpc, linmodel)
            
#             U_data[:,i], Y_data[:,i], Ry_data[:,i], X̂_data[:,i] = u, y, ry, x̂[1:length(x0)]
#             updatestate!(mpc, u, y) # update mpc state estimate
#         end
#         plot_kite(s_plant, i-1)
#         updatestate!(plant, u)  # update plant simulator
#         println("$(dt/t) times realtime at timestep $i. Norm A: $(norm(linmodel.A)). Norm Bu: $(norm(linmodel.Bu)). Norm vsm_jac: $(norm(s_model.prob.ps[sys.vsm_jac]))")
#     end
#     res = SimResult(mpc, U_data, Y_data; Ry_data, X̂_data)
#     return res
# end

# ry = p_model[5](s_model.integrator)
# x̂0 = [
#     x0
#     p_model[5](s_model.integrator)
# ]
# res = sim_adapt!(mpc, model, N, ry, plant, x0, x̂0)
# y_idxs = findall(x -> x != 0.0, Mwt)
# Plots.plot(res; plotx=false, ploty=y_idxs, plotxwithx̂=false, plotu=true, size=(900, 900))
