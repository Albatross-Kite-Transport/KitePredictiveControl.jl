# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

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
integrator = OrdinaryDiffEq.init(prob, FBDF(); dt=Ts, abstol=1e-8, reltol=1e-8, save_on=false, save_everystep=false)
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
    (integrator, _, set_u, _) = p
    reinit!(integrator, x; reinit_dae=false)
    set_u(integrator, u)
    step!(integrator, Ts)
    xnext .= integrator.u # sol.u is the state, called x in the function
    # @info "x: $x u: $u xnext: $xnext"
    return nothing
end
f!(zeros(2), zeros(2), 0.0, nothing, p)
@time f!(zeros(2), zeros(2), 0.0, nothing, p)

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

linmodel = ModelPredictiveControl.linearize(model, x=zeros(2), u=zeros(1))
display(linmodel.A); display(linmodel.Bu)

u = [0.5]
N = 35
# res = sim!(model, N, u, )
# plot(res, plotu=false)


α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)


p_plant = deepcopy(p)
p_plant[1].ps[mtk_model.K] = 1.25 * 1.2
plant = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type); u=vu, x=vx, y=vy)
# res = sim!(estim, N, [0.5], plant=plant, y_noise=[0.5])
# plot(res, plotu=false, plotxwithx̂=true)



Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]
nmpc = NonLinMPC(estim; transcription=SingleShooting(), gradient=ad_type, jacobian=ad_type, Hp, Hc, Mwt, Nwt, Cwt=Inf)
umin, umax = [-1.5], [+1.5]
nmpc = setconstraint!(nmpc; umin, umax)

unset_time_limit_sec(nmpc.optim)
# set_optimizer_attribute(nmpc.optim, "max_iter", 2)
res_ry = sim!(nmpc, 35, [180.0], plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
plot(res_ry)

linmodel = ModelPredictiveControl.linearize(model, x=zeros(2), u=zeros(1))
display(linmodel.A); display(linmodel.Bu)

