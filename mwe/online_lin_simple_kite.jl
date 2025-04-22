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
u_idxs = [1,2,3]

# Initialize model
set_model = deepcopy(load_settings("system_model.yaml"))
set_plant = deepcopy(load_settings("system_plant.yaml"))
s_model = RamAirKite(set_model)
s_plant = RamAirKite(set_plant)
dt = 1/set_model.sample_freq

measure = Measurement()
measure.sphere_pos .= deg2rad.([83.0 83.0; 1.0 -1.0])
KiteModels.init_sim!(s_model, measure; remake=false, adaptive=false)
KiteModels.init_sim!(s_plant, measure; remake=false, adaptive=true)
OrdinaryDiffEq.set_proposed_dt!(s_model.integrator, 0.05dt)
OrdinaryDiffEq.set_proposed_dt!(s_plant.integrator, 0.1dt)
sys = s_model.sys

function stabilize!(s)
    s.integrator.ps[sys.steady] = true
    next_step!(s; dt=10.0, vsm_interval=1)
    s.integrator.ps[sys.steady] = false
end
stabilize!(s_model)
stabilize!(s_plant)

"""
Use quick initialization model to extend the measurements for the kalman filter
"""

# https://github.com/SciML/ModelingToolkit.jl/issues/3552#issuecomment-2817642041
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
struct ModelParams{G1,G2,G3}
    s::RamAirKite
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
    function ModelParams(s)
        x_vec = KiteModels.get_nonstiff_unknowns(s)
        sx_vec = KiteModels.get_stiff_unknowns(s)
        # y_vec = [
        #     [s.sys.tether_length[i] for i in 1:3]
        #     s.sys.elevation
        #     s.sys.azimuth
        # ]
        y_vec = [KiteModels.get_nonstiff_unknowns(s; simple=true); s.sys.angle_of_attack]
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
        
        return new{typeof(get_x), typeof(get_sx), typeof(get_y)}(
            s, s.integrator, set_x, set_ix, set_sx, sx, set_u, 
            get_x, get_sx, get_y, dt, x_vec, sx_vec, y_vec, u_vec,
            sx0, x0, u0
        )
    end
end

p_model = ModelParams(s_model)
p_plant = ModelParams(s_plant)

test_model(p_model)

x_idx = Dict{Num, Int}()
for (idx, sym) in enumerate(p_model.x_vec)
    x_idx[sym] = idx
end
y_idx = Dict{Num, Int}()
for (idx, sym) in enumerate(p_model.y_vec)
    y_idx[sym] = idx
end

nu, nx, ny = length(p_model.u_vec), length(p_model.x_vec), length(p_model.y_vec)
vx, vu, vy = string.(p_model.x_vec), string.(p_model.u_vec), string.(p_model.y_vec)
model = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_model, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
setstate!(model, p_model.x0)
# setop!(model; xop=x0)

N = 100
# res = sim!(model, N, p_model.u0; x_0=x0)
# display(plot(res; plotx=false, ploty=[11,12,13,14,15,16], plotu=false, size=(900, 900)))

nu, nx, ny = length(p_plant.u_vec), length(p_plant.x_vec), length(p_plant.y_vec)
vx, vu, vy = string.(p_plant.x_vec), string.(p_plant.u_vec), string.(p_plant.y_vec)
plant = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)

Hp, Hc, Mwt, Nwt = 30, 2, fill(0.0, model.ny), fill(1.0, model.nu)
Mwt[y_idx[sys.tether_length[1]]] = 1.0
Mwt[y_idx[sys.angle_of_attack]] = rad2deg(1.0)
Mwt[y_idx[sys.kite_pos[2]]] = 0.1

# TODO: linearize on a different core https://www.perplexity.ai/search/using-a-julia-scheduler-run-tw-oKloXmWmSR6YWb47nW_1Gg#0
@time linmodel = ModelPredictiveControl.linearize(model, x=p_model.x0, u=p_model.u0)
display(linmodel.A); display(linmodel.Bu)

# Parameter	Higher Value Means	        Lower Value Means
# Q	        Less trust in model	        More trust in model
# R	        Less trust in measurements	More trust in measurements
σR = fill(0.001, model.ny)
σQ = fill(1.0, model.nx)
σQint_u = fill(0.1, model.nu)
nint_u = fill(1, model.nu)
umin, umax = [-100, -20, -20][u_idxs], [0, 0, 0][u_idxs]
estim = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
mpc = setconstraint!(mpc; umin, umax)

# plot_lin_precision()

function sim_adapt!(mpc, nonlinmodel, N, ry, plant, x0, px0, x̂0, y_step=zeros(ny))
    U_data, Y_data, Ry_data, X̂_data, X_data = 
        zeros(model.nu, N), zeros(model.ny, N), zeros(model.ny, N), zeros(model.nx, N), zeros(plant.nx, N)
    setstate!(plant, px0)
    initstate!(mpc, p_model.u0, plant())
    setstate!(mpc, x̂0)
    for i = 1:N
        t = @elapsed begin
            y = plant() + y_step
            x̂ = preparestate!(mpc, y)
            u = moveinput!(mpc, ry)
            
            reset_p!(p_model, x̂[1:length(p_model.x0)])
            KiteModels.linearize_vsm!(p_model.s)

            # setstate!(model, x0)
            # setstate!(plant, x0)
            
            linmodel = ModelPredictiveControl.linearize(nonlinmodel; u, x=x̂[1:length(x0)])
            setmodel!(mpc, linmodel)
            
            U_data[:,i], Y_data[:,i], Ry_data[:,i], X̂_data[:,i] = u, y, ry, x̂[1:length(x0)]
            updatestate!(mpc, u, y) # update mpc state estimate
        end
        updatestate!(plant, u)  # update plant simulator
        KiteModels.linearize_vsm!(p_plant.s)
        plot_kite(s_plant, i-1)

        X_data[:,i] .= plant.x0
        println("$(dt/t) times realtime at timestep $i. Norm A: $(norm(linmodel.A)). Norm Bu: $(norm(linmodel.Bu)). Norm vsm_jac: $(norm(s_model.prob.ps[sys.vsm_jac]))")
    end
    res = SimResult(mpc, U_data, Y_data; Ry_data, X̂_data, X_data)
    return res
end

ry = p_model.get_y(s_model.integrator)
ry[y_idx[sys.kite_pos[2]]] = 1.0
ry[y_idx[sys.angle_of_attack]] = deg2rad(20)

x̂0 = [
    p_model.x0
    p_model.get_y(s_model.integrator)
]
res = sim_adapt!(mpc, model, N, ry, plant, p_model.x0, p_plant.x0, x̂0)
y_idxs = findall(x -> x != 0.0, Mwt)
Plots.plot(res; plotx=false, ploty=y_idxs, 
    plotxwithx̂=[
        [x_idx[sys.tether_length[i]] for i in 1:3]
        [x_idx[sys.Q_b_w[i]] for i in 1:4]
    ], 
    plotu=true, size=(900, 900)
)

plot_lin_precision()
