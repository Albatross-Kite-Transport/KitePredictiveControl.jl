using ModelPredictiveControl
using ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t, varmap_to_vars
using Serialization
using JuliaSimCompiler
using Plots

const JSC = JuliaSimCompiler

include("mtk_interface.jl")

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

(f_ip, dvs, psym, io_sys) = get_control_function(mtk_model, inputs)
f!, (h!, nu, ny, nx, vu, vy, vx) = generate_f_h(io_sys, inputs, outputs, f_ip, dvs, psym)
Ts = 0.1
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny); u=vu, x=vx, y=vy)

u = [0.5]
N = 35
res = sim!(model, N, u)
plot(res, plotu=false)

α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)

defaults(io_sys)[mtk_model.K] = defaults(io_sys)[mtk_model.K] * 1.25
f_plant!, _ = generate_f_h(io_sys, inputs, outputs, f_ip, dvs, psym)
plant = setname!(NonLinModel(f_plant!, h!, Ts, nu, nx, ny); u=vu, x=vx, y=vy)
res = sim!(estim, N, [0.5], plant=plant, y_noise=[0.5])
plot(res, plotu=false, plotxwithx̂=true)

Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]
nmpc = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
umin, umax = [-1.5], [+1.5]
nmpc = setconstraint!(nmpc; umin, umax)

res_ry = sim!(nmpc, N, [180.0], plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
display(plot(res_ry))

x_0 = zeros(nx)
x̂_0 = zeros(nx + ny)
x_0[ModelingToolkit.variable_index(io_sys, :θ)] = π
x̂_0[ModelingToolkit.variable_index(io_sys, :θ)] = π
res_yd = sim!(nmpc, N, [180.0], plant=plant, x_0=x_0, x̂_0=x̂_0, y_step=[10])
display(plot(res_yd))


nothing