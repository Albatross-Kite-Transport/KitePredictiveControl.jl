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
