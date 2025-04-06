using ModelPredictiveControl, ModelingToolkit, Plots, JuMP, Ipopt, OrdinaryDiffEq
using ModelingToolkit: D_nounits as D, t_nounits as t, setu, setp, getu

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
integrator = OrdinaryDiffEq.init(prob, Tsit5(); dt=Ts, abstol=1e-3, reltol=1e-3, save_on=false, save_everystep=false)
set_x = setu(mtk_model, unknowns(mtk_model))
set_u = setp(mtk_model, [mtk_model.τ])
get_h = getu(mtk_model, [mtk_model.y])
p = (integrator, set_x, set_u, get_h)

function f!(xnext, x, u, _, p)
    (integrator, set_x, set_u, _) = p
    set_x(integrator, x)
    set_u(integrator, u)
    step!(integrator)
    xnext .= integrator.u # integrator.u is the integrator state, called x in the function
    nothing
end
function h!(y, x, _, p)
    (integrator, set_x, _, get_h) = p
    set_x(integrator, x)
    y .= get_h(integrator)
    nothing
end

vx = [string(x) for x in unknowns(mtk_model)]
vu = [string(mtk_model.τ)]
vy = [string(mtk_model.y)]
model = setname!(NonLinModel(f!, h!, Ts, 1, length(unknowns(mtk_model)), 1; p, solver=nothing); u=vu, x=vx, y=vy)


u = [0.5]
N = 35
res = sim!(model, N, u, )
plot(res, plotu=false)


α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)


p_plant = copy(p_model)
p_plant[3] = 1.25*p_model[3]
plant = setname!(NonLinModel(f, h, Ts, nu, nx, ny; p=p_plant); u=vu, x=vx, y=vy)
res = sim!(estim, N, [0.5], plant=plant, y_noise=[0.5])
plot(res, plotu=false, plotxwithx̂=true)


optim = JuMP.Model(Ipopt.Optimizer)
set_optimizer_attribute(optim, "max_iter", 2)
Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]
nmpc = NonLinMPC(estim; transcription=MultipleShooting(), optim, Hp, Hc, Mwt, Nwt, Cwt=Inf)
umin, umax = [-1.5], [+1.5]
nmpc = setconstraint!(nmpc; umin, umax)


res_ry = sim!(nmpc, N, [180.0], plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
plot(res_ry)
