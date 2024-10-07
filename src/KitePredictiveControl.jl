# module KitePredictiveControl

using ModelPredictiveControl
using ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t
using KiteModels, Plots, Serialization, JuliaSimCompiler, OrdinaryDiffEq, MadNLP, JuMP, ForwardDiff, RuntimeGeneratedFunctions, LinearAlgebra

# export run_controller

include("mtk_interface.jl")

set_data_path(joinpath(pwd(), "data"))
if ! @isdefined kite
    kite::KPS4_3L = KPS4_3L(KCU(se("system_3l.yaml")))
    kite.torque_control = true
    KiteModels.init_sim!(kite; prn=true)

end
kite_model, inputs = model!(kite, kite.pos, kite.vel)
kite_model = complete(kite_model)
outputs = [kite_model.flap_angle[i] for i in 1:2]
wanted_outputs = [kite.flap_angle[i] for i in 1:2]
(f_ip, dvs, psym, io_sys) = get_control_function(kite_model, inputs)
f!, (h!, nu, ny, nx, vu, vy, vx) = generate_f_h(io_sys, inputs, outputs, f_ip, dvs, psym)

[defaults(io_sys)[kite_model.pos[j, i]] = kite.pos[i][j] for j in 1:3 for i in 1:kite.num_flap_C-1]
[defaults(io_sys)[kite_model.pos[j, i]] = kite.pos[i][j] for j in 1:3 for i in kite.num_flap_D+1:kite.num_A]
[defaults(io_sys)[kite_model.flap_angle[i]] = kite.flap_angle[i] for i in 1:2]
[defaults(io_sys)[kite_model.tether_length[i]] = kite.tether_lengths[i] for i in 1:3]

x_0 = JuliaSimCompiler.initial_conditions(io_sys, defaults(io_sys), psym)[1]
@show x_0



# function h(x)
#     y = similar(x, 2)
#     h!(y, x, nothing, nothing)
#     return y[2]
# end

# function spline(x)
#     return kite.c_te_interp(x[1], x[2])
# end

# function f_(x)
#     dx = similar(x)
#     f!(dx, x, zeros(3), nothing, nothing)
#     return dx[i]
# end

# """
#     fdiff_derivatives(f::Function) -> Tuple{Function,Function}

# Return a tuple of functions that evaluate the gradient and Hessian of `f` using
# ForwardDiff.jl.
# """
# function fdiff_derivatives(f::Function)
#     function ∇f(g::AbstractVector, x::AbstractVector)
#         ForwardDiff.gradient!(g, f, x)
#         return
#     end
#     function ∇²f(H::AbstractMatrix, x::AbstractVector)
#         h = ForwardDiff.hessian(f, x)
#         for i in 1:length(x), j in 1:i
#             H[i, j] = h[i, j]
#         end
#         return
#     end
#     return ∇f, ∇²f
# end

# dh, ddh = fdiff_derivatives(f_)
# der = zeros(70)
# dh(der, x_0)
# hes = zeros(70, 70)
# ddh(hes, x_0)
# @show norm(hes)
# @show norm(der)

Ts = 1e-3
N = 3
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny, solver=RungeKutta(4; supersample=1000)); u=vu, x=vx, y=vy)

println("sanity check")
u = [100, 100, -100]
@time res = sim!(model, N, u; x_0 = x_0)
display(plot(res, plotu=false))

println("mpc")
max = 100.0
Hp, Hc, Mwt, Nwt, Lwt = 2, 1, fill(0.5, ny), fill(2.5, nu), fill(0, nu)
umin, umax = fill(-max, nu), fill(max, nu)

println("nonlinear")
estim = UnscentedKalmanFilter(model)
nmpc = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt=Inf)
# nmpc = NonLinMPC(model; Hp, Hc)
nmpc = setconstraint!(nmpc; umin=umin, umax=umax)
using JuMP; unset_time_limit_sec(nmpc.optim)
unset_silent(nmpc.optim)

res_ry = sim!(nmpc, N, wanted_outputs.+0.001, plant=model, x_0 = x_0, lastu = [100, 100, -100])
display(plot(res_ry))


# println("linearize")
# # α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1, 1, 1]; σQint_u=[0.1, 0.1, 0.1]
# @time linmodel = ModelPredictiveControl.linearize(model; x=x_0, u=fill(0, nu))
# skf = SteadyKalmanFilter(linmodel; nint_ym=0)
# mpc = LinMPC(linmodel; Hp, Hc, Mwt, Nwt, Cwt=Inf)
# nmpc = setconstraint!(nmpc; umin=umin, umax=umax)
# res = sim!(nmpc, N, wanted_outputs.+0.01, x_0 = x_0, x̂_0 = x̂_0, lastu = [0, 0, -50.0])
# display(plot(res))

nothing

# end