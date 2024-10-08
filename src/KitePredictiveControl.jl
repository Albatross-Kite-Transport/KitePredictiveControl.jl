# module KitePredictiveControl

using ModelPredictiveControl
using ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t
using KiteModels, Plots, Serialization, JuliaSimCompiler, OrdinaryDiffEq, ForwardDiff, RuntimeGeneratedFunctions, LinearAlgebra
using JuMP, DAQP, MadNLP
daqp = Model(DAQP.Optimizer, add_bridges=false)

# export run_controller

include("mtk_interface.jl")

set_data_path(joinpath(pwd(), "data"))
if ! @isdefined kite
    kite::KPS4_3L = KPS4_3L(KCU(se("system_3l.yaml")))
    kite.torque_control = true
    KiteModels.init_sim!(kite; prn=true, torque_control=true)

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

Ts = 1e-2
N = 30
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny, solver=RungeKutta(4; supersample=Int(10/Ts))); u=vu, x=vx, y=vy)

println("sanity check")
u = [100, 100, -100]
@time res = sim!(model, N, u; x_0 = x_0)
display(plot(res, plotu=false))

println("mpc")
max = 500.0
gain = 5e3
Hp, Hc, Mwt, Nwt, Lwt = 2, 1, fill(gain, ny), fill(1/gain, nu), fill(0.0, nu)
umin, umax = fill(-max, nu), fill(max, nu)

println("nonlinear")
estim = UnscentedKalmanFilter(model)
nmpc = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt=Inf)
# nmpc = setconstraint!(nmpc; umin=umin, umax=umax)
using JuMP; unset_time_limit_sec(nmpc.optim)
# unset_silent(nmpc.optim)

x̂_0 = vcat(x_0, wanted_outputs)
@show x̂_0
@time res_ry = sim!(nmpc, N, [0.161, 0.161], plant=model, x_0 = x_0, x̂_0 = x̂_0)
display(plot(res_ry))


# println("linearize")
# α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=0; σQint_u=[0.1, 0.1, 0.1]
# @time linmodel = ModelPredictiveControl.linearize(model; x=x_0, u=fill(0, nu))
# skf = SteadyKalmanFilter(linmodel; nint_ym=0)
# mpc = LinMPC(linmodel; Hp, Hc, Mwt, Nwt, Cwt=Inf)
# nmpc = setconstraint!(nmpc; umin=umin, umax=umax)
# res = sim!(nmpc, N, wanted_outputs.+0.01, x_0 = x_0, x̂_0 = x̂_0, lastu = [0, 0, -50.0])
# display(plot(res))

# println("nonlinear stepping")
# function sim_adapt!(mpc, model, N, ry, plant, x_0, x̂_0, y_step=[0, 0])
#     U_data, Y_data, Ry_data = zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N)
#     setstate!(plant, x_0)
#     initstate!(mpc, [100, 100, -100], plant())
#     setstate!(mpc, x̂_0)
#     for i = 1:N
#         @show i
#         y = plant() + y_step
#         x̂ = preparestate!(mpc, y)
#         @show x̂
#         u = moveinput!(mpc, ry)
#         setmodel!(mpc, model)
#         U_data[:,i], Y_data[:,i], Ry_data[:,i] = u, y, ry
#         updatestate!(mpc, u, y) # update mpc state estimate
#         updatestate!(plant, u)  # update plant simulator
#     end
#     res = SimResult(mpc, U_data, Y_data; Ry_data)
#     return res
# end

# x̂_0 = vcat(x_0, wanted_outputs)
# res_slin = sim_adapt!(nmpc, model, N, wanted_outputs, model, x_0, x̂_0)
# display(plot(res_slin))

nothing

# end