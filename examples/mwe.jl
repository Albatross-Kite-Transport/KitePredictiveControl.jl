using ModelPredictiveControl
using ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t, varmap_to_vars
using JuliaSimCompiler
using Plots
using SeeToDee

const JSC = JuliaSimCompiler

function get_control_function(model, inputs)
    f_ip, dvs, psym, io_sys = ModelingToolkit.generate_control_function(IRSystem(model), inputs)
    return (f_ip, dvs, psym, io_sys)
end

function generate_f_h(io_sys, inputs, outputs, f_ip, dvs, psym, Ts)
    h_ = JuliaSimCompiler.build_explicit_observed_function(io_sys, outputs; inputs = inputs, target = JuliaSimCompiler.JuliaTarget())
    nu = length(inputs)
    ny = length(outputs)
    nx = length(dvs)
    vu = string.(inputs)
    vy = string.(outputs)
    vx = string.(dvs)
    par = JuliaSimCompiler.initial_conditions(io_sys, defaults(io_sys), psym)[2]
    function f_oop(x, u, par, t)
        dx = Vector{Any}(undef, nx)
        f_ip(dx, x, u, par, t)
        return dx
    end
    # fails when using SimpleColloc, but works when using Rk4
    f_disc = SeeToDee.Rk4(f_oop, Ts; supersample = 1)
    f_disc = SimpleColloc(f_oop, Ts, nx, 0, nu)
    SeeToDee.linearize(f_disc, [0.0, 0.0], [0.0], par, 1)
    function f(x, u, _, _)
        return f_disc(x, u, par, 1.0)
    end
    function h!(y, x, _, _)
        h_(y, x, fill(nothing, length(inputs)), par, 1.0)
        nothing
    end
    return f, (h!, nu, ny, nx, vu, vy, vx) # TODO: check on mwe.jl
end

@mtkmodel Pendulum begin
    @parameters begin
        g = 9.8
        L = 0.4
        K = 1.2
        m = 0.3
    end
    @variables begin
        θ(t) = 0.0 # state
        ω(t) = 0.0 # state
        τ(t) = 0.0 # input
        y(t) = 0.0 # output
    end
    @equations begin
        D(θ)    ~ ω
        D(ω)    ~ -g/L*sin(θ) - K/m*ω + τ/m/L^2
        y       ~ θ * 180 / π
    end
end
@named mtk_model = Pendulum()
mtk_model = complete(mtk_model)
inputs, outputs = [mtk_model.τ], [mtk_model.y]

Ts = 0.1
N = 35
(f_ip, dvs, psym, io_sys) = get_control_function(mtk_model, inputs)
f, (h, nu, ny, nx, vu, vy, vx) = generate_f_h(io_sys, inputs, outputs, f_ip, dvs, psym, Ts)
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny, solver=nothing); u=vu, x=vx, y=vy)

println("sanity check")
u = [0.5]
@time res = sim!(model, N, u, x_0 = [0, 0])
display(plot(res, plotu=false))

ModelPredictiveControl.linearize(model)

nothing