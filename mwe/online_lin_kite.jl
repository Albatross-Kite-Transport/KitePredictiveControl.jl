"""
Use a nonlinear MTK model with online linearization and a linear MPC using ModelPredictiveControl.jl
"""

using KiteModels, KiteUtils, LinearAlgebra
using ModelPredictiveControl, ModelingToolkit, Plots, JuMP, Ipopt, OrdinaryDiffEq, FiniteDiff, DifferentiationInterface
using ModelingToolkit: D_nounits as D, t_nounits as t, setu, setp, getu, getp

ad_type = AutoFiniteDiff(relstep=0.01, absstep=0.01)

set_data_path(joinpath(dirname(@__DIR__), "data"))

# Initialize model
set = se("system_ram.yaml")
set.segments = 3
set.quasi_static = true
set.bridle_fracs = [0.088, 0.31, 0.58, 0.93]
set.sample_freq = 20
set.physical_model = "ram"
dt = 1/set.sample_freq

wing = RamAirWing(set; prn=false, n_groups=length(set.bridle_fracs))
aero = BodyAerodynamics([wing])
vsm_solver = Solver(aero; solver_type=NONLIN, atol=2e-8, rtol=2e-8)
point_system = PointMassSystem(set, wing)
s = RamAirKite(set, aero, vsm_solver, point_system)

measure = Measurement()
s.set.abs_tol = 1e-5
s.set.rel_tol = 1e-5

# Initialize at elevation
measure.sphere_pos .= deg2rad.([60.0 60.0; 1.0 -1.0])
KiteModels.init_sim!(s, measure; remake=false)
sys = s.sys

# Stabilize system
s.integrator.ps[sys.steady] = true
next_step!(s; dt=10.0, vsm_interval=1)
s.integrator.ps[sys.steady] = false

solver = FBDF(nlsolve=OrdinaryDiffEq.NLNewton(relax=0.4))
# Function to step simulation with input u
function f(x, u, _, p)
    (s, set_x, set_u, get_x, dt) = p
    global iter += 1
    set_x(s.prob, x)
    set_u(s.prob, u)
    sol = solve(s.prob, solver; dt, abstol=s.set.abs_tol, reltol=s.set.rel_tol, save_on=false, save_everystep=false, save_start=false, verbose=false)
    return get_x(sol)[1]
end

function h(x, _, _)
    return x
end

# Get initial state
iter = 0
x_vec = KiteModels.get_unknowns(s)
inputs = collect(sys.set_values)
set_x = setu(s.integrator, Initial.(x_vec))
set_u = setu(s.integrator, inputs)
get_x = getu(s.integrator, x_vec)
x0 = get_x(s.integrator)
p = (s, set_x, set_u, get_x, dt)

nu, nx, ny = length(inputs), length(x_vec), length(x_vec)
for x in [x0, x0 .+ 0.01]
    for u in [[-50, 0, 0], [-50, -1, -1]]
        for _ in 1:2
            xnext = f(x, u, nothing, p)
            @info "x: $(norm(x)) u: $(norm(u)) xnext: $(norm(xnext))"
        end
    end
end

x_idx = Dict{Num, Int}()
for (idx, sym) in enumerate(x_vec)
    x_idx[sym] = idx
end

vx = string.(x_vec)
vu = string.(inputs)
vy = vx
model = setname!(NonLinModel(f, h, dt, nu, nx, ny; p, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
setstate!(model, x0)
# setop!(model; xop=x0)

umin, umax = [-100, -20, -20], [0, 0, 0]
u = [-50, -5, 0]
N = 10
# res = sim!(model, N, u; x0=x0)
# display(plot(res; plotx=false, ploty=[11,12,13,14,15,16], plotu=false, size=(900, 900)))

α=0.01
σR = fill(0.01, ny)
σQ = fill(0.01, nx)
σQint_u = fill(0.1, nu)
nint_u = fill(1, nu)

p_plant = deepcopy(p)
plant = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
iter = 0
# estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)
# res = sim!(estim, N, [-50, -0.1, -0.1]; x0=x0, plant=plant, y_noise=fill(0.01, ny))
# plot(res; plotx=false, ploty=[11,12,13,14,15,16], plotu=false, plotxwithx̂=false, size=(900, 900))

Hp, Hc, Mwt, Nwt = 20, 2, zeros(ny), fill(0.1, nu)
Mwt[x_idx[sys.ω_b[2]]] = 1.0
Mwt[x_idx[sys.tether_length[1]]] = 1.0
# Mwt[x_idx[sys.tether_vel[2]]] = 0.01
# Mwt[x_idx[sys.tether_vel[3]]] = 0.01
@show Mwt

u0 = [-50, -1, -1]
# TODO: linearize on a different core https://www.perplexity.ai/search/using-a-julia-scheduler-run-tw-oKloXmWmSR6YWb47nW_1Gg#0
@time linmodel = ModelPredictiveControl.linearize(model, x=x0, u=u0)
display(linmodel.A); display(linmodel.Bu)
@show norm(linmodel.A)

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

            # if i%2 == 0
                @time linmodel = ModelPredictiveControl.linearize(nonlinmodel; u, x=x̂[1:length(x0)])
                setmodel!(mpc, linmodel)
                @show norm(linmodel.A)
            # end

            U_data[:,i], Y_data[:,i], Ry_data[:,i], X̂_data[:,i] = u, y, ry, x̂[1:length(x0)]
            updatestate!(mpc, u, y) # update mpc state estimate
        end
        updatestate!(plant, u)  # update plant simulator
        println("$(dt/t) times realtime at timestep $i.")
    end
    res = SimResult(mpc, U_data, Y_data; Ry_data, X̂_data)
    return res
end

ry = x0
x̂0 = [
    x0
    x0
]
res = sim_adapt!(mpc, model, N, ry, plant, x0, x̂0)
y_idxs = [
    [x_idx[sys.ω_b[i]] for i in 1:3]
    x_idx[sys.tether_length[1]]
    x_idx[sys.kite_pos[1]]
]
plot(res; plotx=false, ploty=y_idxs, plotxwithx̂=false, plotu=true, size=(900, 900))
