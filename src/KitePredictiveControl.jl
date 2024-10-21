# module KitePredictiveControl

using ModelPredictiveControl
using ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t
using KiteModels, Plots, Serialization, JuliaSimCompiler, OrdinaryDiffEq, ForwardDiff, RuntimeGeneratedFunctions, LinearAlgebra
using JuMP, DAQP, MadNLP, SeeToDee, NonlinearSolve
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
outputs = vcat(
    vcat(kite_model.flap_angle), 
    reduce(vcat, collect(kite_model.pos[:, 1:kite.num_flap_C])), 
    reduce(vcat, collect(kite_model.pos[:, kite.num_flap_D+1:kite.num_A])),
    vcat(kite_model.tether_length)
)
get_y = ModelingToolkit.getu(kite.integrator.sol, outputs)
initial_outputs = get_y(kite.integrator)
x_0 = deepcopy(kite.integrator.u)

(f_ip, dvs, psym, io_sys) = get_control_function(kite_model, inputs)
f, (h!, nu, ny, nx, vu, vy, vx) = generate_f_h(io_sys, inputs, outputs, f_ip, dvs, psym, x_0)


Ts = 1e-2
N = 300
model = setname!(NonLinModel(f, h!, Ts, nu, nx, ny, solver=nothing); u=vu, x=vx, y=vy)
setstate!(model, x_0)

# println("sanity check")
# u = [100, 100, -100]
# @time res = sim!(model, N, u; x_0 = x_0)
# display(plot(res, plotu=false))

println("mpc")
max = 100.0
gain = 5e3
Hp, Hc = 100, 2
umin, umax = fill(-max, nu), fill(max, nu)
Mwt = fill(0.0, ny)
control_idxs = [findfirst(x -> x == string(kite_model.tether_length[i]), model.yname) for i in 1:3]
Mwt[control_idxs] .= gain
Nwt = fill(1/gain, nu)

println("linearize")
# if !@isdefined linmodel
    @time linmodel = ModelPredictiveControl.linearize(model)
    @time linmodel = ModelPredictiveControl.linearize(model)
    println("sanity check")
    u = [100, 100, -100]
    @time res = sim!(linmodel, N, u; x_0 = x_0)
    display(plot(res; plotx=1:3, ploty=false, plotu=false))
# end

# skf = SteadyKalmanFilter(linmodel; i_ym=1:ny, σQ=fill(1e-3/nx,nx), σR=fill(1e-3,ny), nint_u=0, nint_ym=0)
# skf = SteadyKalmanFilter(linmodel; nint_ym=0, nint_u=0)
estim = KalmanFilter(linmodel)
mpc = LinMPC(estim; Hp, Hc, Mwt=Mwt, Nwt=Nwt, Cwt=Inf)
mpc = setconstraint!(mpc; umin=umin, umax=umax)
@time res = sim!(mpc, N, initial_outputs, x_0 = x_0, lastu = [0, 0, -100.0])
display(plot(res; plotx=1:3, ploty=false, plotu=true))

# xhat should be 86

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