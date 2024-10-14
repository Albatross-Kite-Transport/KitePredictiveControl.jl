# module KitePredictiveControl

using ModelPredictiveControl
using ModelingToolkit
using KiteModels, ControlSystems, Plots, Serialization, OrdinaryDiffEq, RuntimeGeneratedFunctions, LinearAlgebra, SymbolicIndexingInterface
using JuMP, DAQP, MadNLP, SeeToDee, NonlinearSolve, ForwardDiff # solvers
using ControlSystemIdentification, ControlSystemsBase
using ModelingToolkit: variable_index as idx
# import RobustAndOptimalControl: named_ss

# export run_controller

include("mtk_interface.jl")

set_data_path(joinpath(pwd(), "data"))
if ! @isdefined kite
    kite::KPS4_3L = KPS4_3L(KCU(se("system_3l.yaml")))
    kite.torque_control = true
    KiteModels.init_sim!(kite; prn=true, torque_control=true)

    kite_model, inputs = model!(kite, kite.pos, kite.vel)
    kite_model = complete(kite_model)
end
outputs = vcat(
    vcat(kite_model.flap_angle), 
    reduce(vcat, collect(kite_model.pos[:, 1:kite.num_flap_C-1])), 
    reduce(vcat, collect(kite_model.pos[:, kite.num_flap_D+1:kite.num_A])),
    vcat(kite_model.tether_length),
    kite_model.winch_force[3]
)
initial_outputs = vcat(
    vcat(kite.flap_angle), 
    reduce(vcat, kite.pos[1:kite.num_flap_C-1]), 
    reduce(vcat, kite.pos[kite.num_flap_D+1:kite.num_A]),
    vcat(kite.tether_lengths),
    norm.(kite.winch_forces)[3]
)

[defaults(kite_model)[kite_model.pos[j, i]] = kite.pos[i][j] for j in 1:3 for i in 1:kite.num_flap_C-1]
[defaults(kite_model)[kite_model.pos[j, i]] = kite.pos[i][j] for j in 1:3 for i in kite.num_flap_D+1:kite.num_A]
[defaults(kite_model)[kite_model.flap_angle[i]] = kite.flap_angle[i] for i in 1:2]
[defaults(kite_model)[kite_model.tether_length[i]] = kite.tether_lengths[i] for i in 1:3]

lin_fun, sys = ModelingToolkit.linearization_function(kite_model, inputs, outputs)
Ts = 0.1
N = 400

(; A, B, C, D) = ModelingToolkit.linearize(sys, lin_fun; t=1.0, op = defaults(sys));
css = ss(A, B, C, D)
dss = c2d(css, Ts)
model = LinModel(dss.A, dss.B, dss.C, dss.B[:, end+1:end], dss.D[:, end+1:end], Ts)
x_0 = ModelingToolkit.varmap_to_vars(defaults(sys), unknowns(sys))
setstate!(model, x_0)

model.uname .= string.(inputs)
model.yname .= string.(outputs)
model.xname .= string.(unknowns(sys))
 
println("linear mpc")
Hp, Hc = 100, 20
umin, umax = [-40, -40, -200], [0, 0, 0]
output_idxs = vcat(
    idx(sys, sys.pos[2, kite.num_A]),
    model.ny,
)
observed_idxs = vcat(
    idx(sys, sys.pos[2, kite.num_A]),
    idx(sys, sys.tether_length[3]),
    model.nx + model.ny
)

Mwt = fill(0.0, model.ny)
Mwt[output_idxs] .= 1.0
Nwt = fill(10, model.nu)

# println("sanity check")
# u = [-10, -10, -500]
# res = sim!(model, N, u; x_0 = x_0)
# display(plot(res; plotx=tether_idxs, ploty=false, plotu=false))

σR = fill(0.01, model.ny)
σR[model.ny] = 1.0
estim = SteadyKalmanFilter(model; nint_u=fill(1, model.nu), σQint_u=fill(0.1, model.nu), σQ = fill(0.3, model.nx), σR = σR) # sigma q important!
mpc = LinMPC(estim; Hp, Hc, Mwt=Mwt, Nwt=Nwt, Cwt=1e5)

x̂max = fill(Inf, model.nx+model.ny)
x̂min = fill(-Inf, model.nx+model.ny)
x̂max[tether_idx] = x_0[tether_idx] + 10.0
x̂min[tether_idx] = x_0[tether_idx] - 10.0
x̂min[end] = -500.0
x̂max[end] = 500

mpc = setconstraint!(mpc; umin=umin, umax=umax, x̂max = x̂max, x̂min = x̂min)
@time res = sim!(mpc, N, initial_outputs .+ 0.1, x_0 = x_0, lastu = [0, 0, 0]) # plant=model
display(plot(res; plotx=false, ploty=output_idxs, plotu=true, plotxwithx̂=observed_idxs))
# println("maximum force: ", maximum([force(res.X_data[:, i], fill(0, length(inputs)), par, 1.0) for i in 1:N]))




# println("stepping linear mpc with nonlinear plant")

# estim = KalmanFilter(model; σQ, σR, nint_u, σQint_u)
# mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf, optim=daqp)
# mpc = setconstraint!(mpc; umin, umax)
# function sim_adapt!(mpc, kite_model, N, ry, plant, x_0, x̂_0, y_step=[0])
#     U_data, Y_data, Ry_data = zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N)
#     setstate!(plant, x_0)
#     initstate!(mpc, [0], plant())
#     setstate!(mpc, x̂_0)
#     for i = 1:N
#         y = plant() + y_step
#         x̂ = preparestate!(mpc, y)
#         u = moveinput!(mpc, ry)
#         model = linearize(kite_model; u, x=x̂[1:2])
#         (; A, B, C, D) = ModelingToolkit.linearize(sys, lin_fun; t=1.0, op = );
#         css = ss(A, B, C, D)
#         dss = c2d(css, Ts)
#         model = LinModel(dss.A, dss.B, dss.C, dss.B[:, end+1:end], dss.D[:, end+1:end], Ts)
#         setmodel!(mpc, model)
#         U_data[:,i], Y_data[:,i], Ry_data[:,i] = u, y, ry
#         updatestate!(mpc, u, y) # update mpc state estimate
#         updatestate!(plant, u)  # update plant simulator
#     end
#     res = SimResult(mpc, U_data, Y_data; Ry_data)
#     return res
# end
# x_0 = [0, 0]; x̂_0 = [0, 0, 0]; ry = [180]
# res_slin = sim_adapt!(mpc, kite_model, N, ry, plant, x_0, x̂_0)
# plot(res_slin)

nothing

# end