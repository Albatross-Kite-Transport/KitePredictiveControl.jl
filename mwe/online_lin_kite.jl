"""
Use a nonlinear MTK model with online linearization and a linear MPC using ModelPredictiveControl.jl
"""

using KiteModels, KiteUtils, LinearAlgebra
using ModelPredictiveControl, ModelingToolkit, Plots, JuMP, Ipopt, OrdinaryDiffEq, FiniteDiff, DifferentiationInterface
using ModelingToolkit: D_nounits as D, t_nounits as t, setu, setp, getu, getp

ad_type = AutoFiniteDiff(relstep=1.0, absstep=1.0)
dt = 0.05

set_data_path(joinpath(dirname(@__DIR__), "data"))
set = se("system_ram.yaml")
set_values = [-50, -1.1, -1.1]
if !@isdefined s
    wing = RamAirWing(set)
    aero = BodyAerodynamics([wing])
    vsm_solver = Solver(aero; solver_type=NONLIN, atol=1e-8, rtol=1e-8)
    point_system = PointMassSystem(set, wing)
    s = RamAirKite(set, aero, vsm_solver, point_system)
    measure = Measurement()
end
if !ispath(joinpath(get_data_path(), "prob.bin"))
    KiteModels.init_sim!(s, measure)
end
measure.sphere_pos .= deg2rad.([60.0 60.0; 1.0 -1.0])
if !ispath(joinpath(get_data_path(), "prob.bin"))
    KiteModels.init_sim!(s, measure)
end
@time KiteModels.reinit!(s, measure; reload=true)
sys = s.sys
s.integrator.ps[sys.steady] = true
next_step!(s; dt=10.0, vsm_interval=1)
s.integrator.ps[sys.steady] = false

prob = s.prob
KiteModels.reinit!(s, measure)
mtk_model = s.sys
inputs = [mtk_model.set_values[i] for i in 1:3]
x_vec = KiteModels.get_nonstiff_unknowns(s)
sx_vec = KiteModels.get_stiff_unknowns(s)
set_x = setu(s.integrator, x_vec)
set_sx = setu(s.integrator, sx_vec)
get_x = getu(s.integrator, x_vec)
get_sx = getu(s.integrator, sx_vec)
x0 = get_x(s.integrator)
sx0 = get_sx(s.integrator)
p = (s, sx0, set_x, set_sx, get_x, dt)

function f!(xnext, x, u, _, p)
    (s, stiff_x, set_x, set_sx, get_x, dt) = p
    @show x u
    set_x(s.integrator, x)
    set_sx(s.integrator, stiff_x)
    OrdinaryDiffEq.reinit!(s.integrator, s.integrator.u; reinit_dae=false)
    next_step!(s, u; dt, vsm_interval=0)
    xnext .= get_x(s.integrator)
    return nothing
end

function h!(y, x, _, _)
    y .= x
    nothing
end

nu, nx, ny = length(inputs), length(x_vec), length(x_vec)
# xnext = zeros(nx)
# for x in [x0, x0 .+ 0.001]
#     for u in [[-50, 0, 0], [-50, -1, -1]]
#         for _ in 1:2
#             f!(xnext, x, u, nothing, p)
#             @info "x: $(norm(x)) u: $(norm(u)) xnext: $(norm(xnext))"
#         end
#     end
# end

vx = string.(x_vec)
vu = string.(inputs)
vy = vx
model = setname!(NonLinModel(f!, h!, dt, nu, nx, ny; p, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
setstate!(model, x0)
# setop!(model; xop=x0)

umin, umax = [-100, -10, -10], [0, 10, 10]
u = [-50, -5, 0]
N = 5
res = sim!(model, N, u; x_0=x0)
display(plot(res; plotx=false, ploty=[11,12,13,14,15,16], plotu=false, size=(900, 900)))

α=0.01
σR = fill(0.01, ny)
σQ = fill(0.01, nx)
σQint_u = fill(0.1, nu)
nint_u = fill(1, nu)
estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)

p_plant = deepcopy(p)
plant = setname!(NonLinModel(f!, h!, dt, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
res = sim!(estim, N, [-50, -0.1, -0.1]; x_0=x0, plant=plant, y_noise=fill(0.01, ny))
plot(res; plotx=false, ploty=[11,12,13,14,15,16], plotu=false, plotxwithx̂=false, size=(900, 900))

# Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]

# linmodel = ModelPredictiveControl.linearize(model, x=[0, 0], u=[0])
# display(linmodel.A); display(linmodel.Bu)

# kf = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
# mpc3 = LinMPC(kf; Hp, Hc, Mwt, Nwt, Cwt=Inf)
# mpc3 = setconstraint!(mpc3; umin, umax)

# function sim_adapt!(mpc, nonlinmodel, N, ry, plant, x_0, x̂_0, y_step=[0])
#     U_data, Y_data, Ry_data = zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N)
#     setstate!(plant, x_0)
#     initstate!(mpc, [0], plant())
#     setstate!(mpc, x̂_0)
#     for i = 1:N
#         y = plant() + y_step
#         x̂ = preparestate!(mpc, y)
#         @time u = moveinput!(mpc, ry)
#         linmodel = ModelPredictiveControl.linearize(nonlinmodel; u, x=x̂[1:2])
#         setmodel!(mpc, linmodel)
#         U_data[:,i], Y_data[:,i], Ry_data[:,i] = u, y, ry
#         updatestate!(mpc, u, y) # update mpc state estimate
#         updatestate!(plant, u)  # update plant simulator
#     end
#     res = SimResult(mpc, U_data, Y_data; Ry_data)
#     return res
# end

# x_0 = [0, 0]; x̂_0 = [0, 0, 0]; ry = [180]
# res_slin = sim_adapt!(mpc3, model, N, ry, plant, x_0, x̂_0)
# plot(res_slin)
