using ModelPredictiveControl, ModelingToolkit, Plots, JuMP, Ipopt, OrdinaryDiffEq, FiniteDiff, DifferentiationInterface, SimpleDiffEq
using ModelingToolkit: D_nounits as D, t_nounits as t, setu, setp, getu, getp

ad_type = AutoFiniteDiff(relstep=1e-2, absstep=1e-2)

Ts = 0.1
@mtkmodel Pendulum begin
    @parameters begin
        g = 9.8
        L = 0.4
        K = 1.2
        m = 0.3
        τ = 0.0 # input
    end
    @variables begin
        θ(t) = 0.0 # state
        ω(t) = 0.0 # state
        y(t) # output
    end
    @equations begin
        D(θ)    ~ ω
        D(ω)    ~ -g/L*sin(θ) - K/m*ω + τ/m/L^2
        y       ~ θ * 180 / π
    end
end

@mtkbuild mtk_model = Pendulum()
prob = ODEProblem(mtk_model, nothing, (0.0, Ts))
integrator = OrdinaryDiffEq.init(prob, FBDF(); dt=Ts, abstol=1e-6, reltol=1e-6, save_on=false, save_everystep=false)
set_x = setu(mtk_model, unknowns(mtk_model))
get_x = getu(mtk_model, unknowns(mtk_model))
set_u = setp(mtk_model, [mtk_model.τ])
get_u = getp(mtk_model, [mtk_model.τ])
get_h = getu(mtk_model, [mtk_model.y])
p = (integrator, set_x, set_u, get_h)

function f!(xnext, x, u, _, p)
    if any(isnan.(x)) || any(isnan.(u))
        xnext .= NaN
        return nothing
    end
    (integrator, set_x, set_u, _) = p
    set_t!(integrator, 0.0)
    set_proposed_dt!(integrator, Ts)
    set_x(integrator, x)
    set_u(integrator, u)
    step!(integrator, Ts)
    xnext .= integrator.u # sol.u is the state, called x in the function
    return nothing
end
f!(zeros(2), zeros(2), 0.0, nothing, p)
@time f!(zeros(2), zeros(2), 0.0, nothing, p)

xnext = zeros(2)
for x in [zeros(2), ones(2)]
    for u in [[0.0], [1.0]]
        for _ in 1:2
            f!(xnext, x, u, nothing, p)
            @info "x: $x u: $u xnext: $xnext"
        end
    end
end

function h!(y, x, _, p)
    (integrator, set_x, _, get_h) = p
    set_x(integrator, x)
    y .= get_h(integrator)
    nothing
end

nu, nx, ny = 1, 2, 1
vx = [string(x) for x in unknowns(mtk_model)]
vu = [string(mtk_model.τ)]
vy = [string(mtk_model.y)]
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)

linmodel = ModelPredictiveControl.linearize(model, x=[0, 0], u=[0])
@info "Linearized model: " linmodel.A linmodel.Bu

# using JuMP, DAQP
# Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]
# α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
# umin, umax = [-1.5], [+1.5]
# N = 35
# daqp = Model(DAQP.Optimizer, add_bridges=false)
# kf = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
# mpc = LinMPC(kf; Hp, Hc, Mwt, Nwt, Cwt=Inf, optim=daqp)
# mpc = setconstraint!(mpc; umin, umax)

# p_plant = deepcopy(p)
# @assert p[1] != p_plant[1]
# p_plant[1].ps[mtk_model.K] = 1.25 * 1.2
# plant = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)

# function sim_adapt!(mpc, nonlinmodel, N, ry, plant, x_0, x̂_0, y_step=[0])
#     println("iterate")
#     U_data, Y_data, Ry_data = zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N)
#     setstate!(plant, x_0)
#     initstate!(mpc, [0], plant())
#     setstate!(mpc, x̂_0)
#     for i = 1:N
#         y = plant() + y_step
#         x̂ = preparestate!(mpc, y)
#         u = moveinput!(mpc, ry)
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
# res_slin = sim_adapt!(mpc, model, N, ry, plant, x_0, x̂_0)
# plot(res_slin)


