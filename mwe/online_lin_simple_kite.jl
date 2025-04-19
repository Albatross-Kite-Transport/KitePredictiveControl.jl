"""
Use a nonlinear MTK model with online linearization and a linear MPC using ModelPredictiveControl.jl
"""

using KiteModels, KiteUtils, LinearAlgebra
using ModelPredictiveControl, ModelingToolkit, Plots, JuMP, Ipopt, OrdinaryDiffEq, FiniteDiff, DifferentiationInterface
using ModelingToolkit: D_nounits as D, t_nounits as t, setu, setp, getu, getp
using ControlPlots

include(joinpath(@__DIR__, "plotting.jl"))

ad_type = AutoFiniteDiff()

set_data_path(joinpath(dirname(@__DIR__), "data"))

# Initialize model
set_model = deepcopy(se("system_model.yaml"))
dt = 1/set_model.sample_freq
s_model = RamAirKite(set_model)

set_plant = deepcopy(se("system_plant.yaml"))
s_plant = RamAirKite(set_plant)

measure = Measurement()
measure.set_values .= [-55, -4.0, -4.0]  # Set values of the torques of the three winches. [Nm]
set_values = measure.set_values

# Initialize at elevation
measure.sphere_pos .= deg2rad.([83.0 83.0; 1.0 -1.0])
KiteModels.init_sim!(s_model, measure; remake=false)
KiteModels.init_sim!(s_plant, measure; remake=false)
sys = s_model.sys

function stabilize!(s)
    # Stabilize system
    s.integrator.ps[s.sys.steady] = true
    next_step!(s; dt=10.0, vsm_interval=1)
    s.integrator.ps[s.sys.steady] = false
    @info "Init"
    for i in 1:10
        set_values = -s.set.drum_radius * s.integrator[s.sys.winch_force]
        KiteModels.next_step!(s, set_values; dt)
        # plot_kite(s, i-1)
    end
end
stabilize!(s_model)
stabilize!(s_plant)

function linearize_vsm!(s::RamAirKite)
    y = s.get_y(s.integrator.u)
    jac, x = VortexStepMethod.linearize(
        s.vsm_solver, 
        s.aero, 
        y;
        va_idxs=1:3, 
        omega_idxs=4:6,
        theta_idxs=7:6+length(s.point_system.groups),
        moment_frac=s.bridle_fracs[s.point_system.groups[1].fixed_index])
    set_vsm(s.prob, [x, y, jac])
    nothing
end

solver = FBDF(nlsolve=OrdinaryDiffEq.NLNewton(relax=0.4))
# Function to step simulation with input u
function f(x, u, _, p)
    (s, set_x, set_u, get_x, _, dt, _) = p
    set_x(s.prob, x)
    set_u(s.prob, u)
    sol = solve(s.prob, solver; dt, abstol=s.set.abs_tol, reltol=s.set.rel_tol, save_on=false, save_everystep=false, save_start=false, verbose=false)
    return get_x(sol)[1]
end

function f_plant(x, u, _, p)
    (s, _, _, get_x, _, dt, _) = p
    next_step!(s, u; dt)
    return get_x(s.integrator)
end

function h(x, _, p)
    (s, _, _, _, get_y, _, set_xh) = p
    set_xh(s.integrator, x)
    y = get_y(s.integrator)
    return y
end

# Get initial state
iter = 0
function get_p(s)
    x_vec = KiteModels.get_unknowns(s)
    y_vec = [
        collect(s.sys.tether_vel)
    ]
    inputs = collect(s.sys.set_values)
    set_x = setu(s.integrator, Initial.(x_vec))
    set_xh = setu(s.integrator, x_vec)
    set_u = setu(s.integrator, inputs)
    get_x = getu(s.integrator, x_vec)
    get_y = getu(s.integrator, y_vec)
    x0 = get_x(s.integrator)
    u0 = -s.set.drum_radius * s.integrator[sys.winch_force]
    return (s, set_x, set_u, get_x, get_y, dt, set_xh), x_vec, y_vec, x0, u0, inputs
end
p_model, x_vec, y_vec, x0, u0, inputs = get_p(s_model)
p_plant, _, _, _, _, _ = get_p(s_plant)

nu, nx, ny = length(inputs), length(x_vec), length(y_vec)
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

y_idx = Dict{Num, Int}()
for (idx, sym) in enumerate(y_vec)
    y_idx[sym] = idx
end

vx = string.(x_vec)
vu = string.(inputs)
vy = string.(y_vec)
model = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_model, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
setstate!(model, x0)
# setop!(model; xop=x0)

u = [-50, -5, 0]
N = 90
# res = sim!(model, N, u; x_0=x0)
# display(plot(res; plotx=false, ploty=[11,12,13,14,15,16], plotu=false, size=(900, 900)))

α=0.01
σR = fill(0.01, ny)
σQ = fill(0.01, nx)
σQint_u = fill(0.1, nu)
nint_u = fill(1, nu)

plant = setname!(NonLinModel(f_plant, h, dt, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
s_plant = p_plant[1]
iter = 0
# estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)
# res = sim!(estim, N, [-50, -0.1, -0.1]; x_0=x0, plant=plant, y_noise=fill(0.01, ny))
# plot(res; plotx=false, ploty=[11,12,13,14,15,16,17], plotu=false, plotxwithx̂=false, size=(900, 900))

Hp, Hc, Mwt, Nwt = 20, 2, fill(1.0, ny), fill(1.0, nu)

# TODO: linearize on a different core https://www.perplexity.ai/search/using-a-julia-scheduler-run-tw-oKloXmWmSR6YWb47nW_1Gg#0
@time linmodel = ModelPredictiveControl.linearize(model, x=x0, u=u0)
display(linmodel.A); display(linmodel.Bu)

umin, umax = [-100, -20, -20], [0, 0, 0]
ymin, ymax = fill(-Inf, ny), fill(Inf, ny)
[ymin[y_idx[sys.tether_acc[i]]] = -10.0 for i in 1:3]
[ymax[y_idx[sys.tether_acc[i]]] = 10.0 for i in 1:3]
Δumin, Δumax = [-10,-1,-1], [10,1,1]

estim = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
mpc = setconstraint!(mpc; umin, umax)

function sim_adapt!(mpc, nonlinmodel, N, ry, plant, x0, x̂_0, y_step=zeros(ny))
    U_data, Y_data, Ry_data, X̂_data = zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N), zeros(plant.nx, N)
    setstate!(plant, x0)
    initstate!(mpc, u0, plant())
    setstate!(mpc, x̂_0)
    for i = 1:N
        t = @elapsed begin
            
            y = plant() + y_step
            x̂ = preparestate!(mpc, y)
            u = moveinput!(mpc, ry)
            
            vsm_y = s_plant.get_y(s_plant.integrator) # TODO: use model
            vsm_jac, vsm_x = VortexStepMethod.linearize(
                s_model.vsm_solver, 
                s_model.aero, 
                vsm_y;
                va_idxs=1:3, 
                omega_idxs=4:6,
                theta_idxs=7:6+length(s_model.point_system.groups),
                moment_frac=s_model.bridle_fracs[s_model.point_system.groups[1].fixed_index])
            s_model.set_vsm(s_model.prob, [vsm_x, vsm_y, vsm_jac])
            s_plant.set_vsm(s_plant.integrator, [vsm_x, vsm_y, vsm_jac])

            linmodel = ModelPredictiveControl.linearize(nonlinmodel; u, x=x̂[1:length(x0)])
            setmodel!(mpc, linmodel)
            
            U_data[:,i], Y_data[:,i], Ry_data[:,i], X̂_data[:,i] = u, y, ry, x̂[1:length(x0)]
            updatestate!(mpc, u, y) # update mpc state estimate
        end
        plot_kite(s_plant, i-1)
        updatestate!(plant, u)  # update plant simulator
        println("$(dt/t) times realtime at timestep $i. Norm A: $(norm(linmodel.A)).")
    end
    res = SimResult(mpc, U_data, Y_data; Ry_data, X̂_data)
    return res
end

ry = p_model[5](s_model.integrator)
x̂0 = [
    x0
    p_model[5](s_model.integrator)
]
res = sim_adapt!(mpc, model, N, ry, plant, x0, x̂0)
y_idxs = findall(x -> x != 0.0, Mwt)
Plots.plot(res; plotx=false, ploty=y_idxs, plotxwithx̂=false, plotu=true, size=(900, 900))
