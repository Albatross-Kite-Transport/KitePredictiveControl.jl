"""
Use a nonlinear MTK model with online linearization and a linear MPC using ModelPredictiveControl.jl
"""

using KiteModels
using ModelPredictiveControl, ModelingToolkit, Plots, JuMP, Ipopt, OrdinaryDiffEq, FiniteDiff, DifferentiationInterface
using ModelingToolkit: D_nounits as D, t_nounits as t, setu, setp, getu, getp

ad_type = AutoFiniteDiff(relstep=0.01, absstep=0.01)
Ts = 0.1

set = se("system_ram.yaml")
set.segments = 2
set_values = [-50, -1.1, -1.1]

if !@isdefined s
    wing = RamAirWing(set)
    aero = BodyAerodynamics([wing])
    vsm_solver = Solver(aero; solver_type=NONLIN, atol=1e-8, rtol=1e-8)
    point_system = PointMassSystem(set, wing)
    s = RamAirKite(set, aero, vsm_solver, point_system)

    measure = Measurement()
end
s.set.abs_tol = 1e-5
s.set.rel_tol = 1e-3

if !ispath(joinpath(get_data_path(), "prob.bin"))
    KiteModels.init_sim!(s, measure)
end
measure.sphere_pos .= deg2rad.([50.0 50.0; 1.0 -1.0])
@time KiteModels.reinit!(s, measure)


prob = s.prob
integrator = OrdinaryDiffEq.init(prob, FBDF(); dt=Ts, abstol=1e-8, reltol=1e-8, save_on=false, save_everystep=false)
set_x = setu(mtk_model, unknowns(mtk_model))
get_x = getu(mtk_model, unknowns(mtk_model))
set_u = setp(mtk_model, [mtk_model.τ])
get_u = getp(mtk_model, [mtk_model.τ])
get_h = getu(mtk_model, [mtk_model.y])
p = (integrator, set_x, set_u, get_h)

function f!(xnext, x, u, _, p)
    # if any(isnan.(x)) || any(isnan.(u))
    #     xnext .= NaN
    #     return nothing
    # end
    (integrator, _, set_u, _) = p
    reinit!(integrator, x; reinit_dae=false)
    set_u(integrator, u)
    step!(integrator, Ts)
    xnext .= integrator.u # sol.u is the state, called x in the function
    # @info "x: $x u: $u xnext: $xnext"
    return nothing
end

function h!(y, x, _, p)
    (integrator, set_x, _, get_h) = p
    set_x(integrator, x)
    y .= get_h(integrator)
    nothing
end

nu, nx, ny, Ts = 1, 2, 1, 0.1
vx = [string(x) for x in unknowns(mtk_model)]
vu = [string(mtk_model.τ)]
vy = [string(mtk_model.y)]
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)

umin, umax = [-1.5], [+1.5]
u = [0.5]
N = 35

α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)

p_plant = deepcopy(p)
p_plant[1].ps[mtk_model.K] = 1.25 * 1.2
plant = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
res = sim!(estim, N, [0.5], plant=plant, y_noise=[0.5])
plot(res, plotu=false, plotxwithx̂=true)

Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]

linmodel = ModelPredictiveControl.linearize(model, x=[0, 0], u=[0])
display(linmodel.A); display(linmodel.Bu)

kf = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
mpc3 = LinMPC(kf; Hp, Hc, Mwt, Nwt, Cwt=Inf)
mpc3 = setconstraint!(mpc3; umin, umax)

function sim_adapt!(mpc, nonlinmodel, N, ry, plant, x_0, x̂_0, y_step=[0])
    U_data, Y_data, Ry_data = zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N)
    setstate!(plant, x_0)
    initstate!(mpc, [0], plant())
    setstate!(mpc, x̂_0)
    for i = 1:N
        y = plant() + y_step
        x̂ = preparestate!(mpc, y)
        @time u = moveinput!(mpc, ry)
        linmodel = ModelPredictiveControl.linearize(nonlinmodel; u, x=x̂[1:2])
        setmodel!(mpc, linmodel)
        U_data[:,i], Y_data[:,i], Ry_data[:,i] = u, y, ry
        updatestate!(mpc, u, y) # update mpc state estimate
        updatestate!(plant, u)  # update plant simulator
    end
    res = SimResult(mpc, U_data, Y_data; Ry_data)
    return res
end

x_0 = [0, 0]; x̂_0 = [0, 0, 0]; ry = [180]
res_slin = sim_adapt!(mpc3, model, N, ry, plant, x_0, x̂_0)
plot(res_slin)
