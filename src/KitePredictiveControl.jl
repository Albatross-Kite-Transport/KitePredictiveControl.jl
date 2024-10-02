# module KitePredictiveControl

using ModelPredictiveControl
using ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t
using KiteModels
using Plots
using Serialization
using JuliaSimCompiler
using OrdinaryDiffEq

# export run_controller

include("mtk_interface.jl")

set_data_path(joinpath(pwd(), "data"))
if ! @isdefined kite; kite::KPS4_3L = KPS4_3L(KCU(se("system_3l.yaml"))); end
kite.torque_control = true
KiteModels.init_sim!(kite; prn=true)

kite_model, inputs = model!(kite, kite.pos, kite.vel)
kite_model = complete(kite_model)
outputs = [kite_model.pos[i, kite.num_A] for i in 1:3]
wanted_outputs = [kite.pos[kite.num_A][i] for i in 1:3]
@show wanted_outputs

(f_ip, dvs, psym, io_sys) = get_control_function(kite_model, inputs)
f!, (h!, nu, ny, nx, vu, vy, vx) = generate_f_h(io_sys, inputs, outputs, f_ip, dvs, psym)
Ts = 0.1
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny, solver=RungeKutta(4; supersample=5000)); u=vu, x=vx, y=vy)

println("sanity check")
u = [0.0, 0.0, 0.0]
N = 10
x_0 = JuliaSimCompiler.initial_conditions(io_sys, defaults(io_sys), psym)[1]
@time res = sim!(model, N, u; x_0 = x_0)
display(plot(res, plotu=false))

println("mpc")
max = 10
Hp, Hc, Mwt, Nwt = 10, 2, [0.5], [2.5]
nmpc = NonLinMPC(model; Hp, Hc, Cwt=Inf)
umin, umax = fill(-max, 3), fill(max, 3)
nmpc = setconstraint!(nmpc; umin=umin, umax=umax)

x̂_0 = zeros(length(x_0) + length(wanted_outputs))
f!(x̂_0, x_0, zeros(3), nothing, nothing)
res_ry = sim!(nmpc, 20, wanted_outputs, x_0 = x_0, x̂_0 = x̂_0)
display(plot(res_ry))


# println("linearize")
# @time ModelPredictiveControl.linearize(model; x=x_0, u=[0,0,0])
nothing

# end