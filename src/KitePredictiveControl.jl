# module KitePredictiveControl

using ModelPredictiveControl
using ModelingToolkit
using KiteModels, ControlSystems, Plots, Serialization, OrdinaryDiffEq, RuntimeGeneratedFunctions, LinearAlgebra, SymbolicIndexingInterface
using JuMP, DAQP, MadNLP, SeeToDee, NonlinearSolve, ForwardDiff # solvers
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
    reduce(vcat, collect(kite_model.pos[:, 1:kite.num_flap_C-1])), 
    reduce(vcat, collect(kite_model.pos[:, kite.num_flap_D+1:kite.num_A])),
    vcat(kite_model.tether_length)
)
initial_outputs = vcat(
    vcat(kite.flap_angle), 
    reduce(vcat, kite.pos[1:kite.num_flap_C-1]), 
    reduce(vcat, kite.pos[kite.num_flap_D+1:kite.num_A]),
    vcat(kite.tether_lengths)
)

[defaults(kite_model)[kite_model.pos[j, i]] = kite.pos[i][j] for j in 1:3 for i in 1:kite.num_flap_C-1]
[defaults(kite_model)[kite_model.pos[j, i]] = kite.pos[i][j] for j in 1:3 for i in kite.num_flap_D+1:kite.num_A]
[defaults(kite_model)[kite_model.flap_angle[i]] = kite.flap_angle[i] for i in 1:2]
[defaults(kite_model)[kite_model.tether_length[i]] = kite.tether_lengths[i] for i in 1:3]

Ts = 0.1
N = 10
@time (; A, B, C, D), simple_sys = ModelingToolkit.linearize(kite_model, inputs, outputs; t=1.0, op = defaults(kite_model));
x_0 = ModelingToolkit.varmap_to_vars(defaults(simple_sys), unknowns(simple_sys))
u_0 = ModelingToolkit.varmap_to_vars(defaults(simple_sys), inputs)
y_0 = ModelingToolkit.varmap_to_vars(defaults(simple_sys), outputs)
model = LinModel(A, B, C, B[:, end+1:end], D[:, end+1:end], Ts)
setstate!(model, x_0)
setop!(model, xop=x_0, yop=y_0, uop=u_0)
model.uname .= string.(inputs)
model.yname .= string.(outputs)
model.xname .= string.(unknowns(kite.simple_sys))
 
tether_idxs = variable_index(kite.simple_sys, :tether_length)

println("linear mpc")
Hp, Hc = 40, 10
umin, umax = [-50, -50, -200], [0, 0, 0]
Mwt = fill(0, model.ny)
output_idxs = [findfirst(x -> x == string(kite_model.tether_length[i]), model.yname) for i in 1:3]
ratio = norm.(kite.winch_forces)[3] / norm.(kite.winch_forces)[1]
Mwt[output_idxs] .= [1, 1, round(ratio)]
Nwt = fill(0.001, model.nu)

println("sanity check")
u = zeros(3)
res = sim!(model, N, [100, 100, 100]; x_0 = x_0)
display(plot(res; plotx=tether_idxs, ploty=false, plotu=false))

# estim = SteadyKalmanFilter(model; nint_u=fill(1, model.nu), σQint_u=fill(0.1, model.nu), σQ = fill(0.1, model.nx), σR = fill(0.1, model.ny)) # sigma q important!
# mpc = NonLinMPC(estim; Hp, Hc, Mwt=Mwt, Nwt=Nwt, Cwt=Inf)
# mpc = setconstraint!(mpc; umin=umin, umax=umax)
# @time res = sim!(mpc, N, initial_outputs.-1.0, x_0 = x_0, lastu = [0, 0, 0]) # plant=model
# display(plot(res; plotx=false, ploty=output_idxs, plotu=true, plotxwithx̂=[2, 3, 1]))
# # display(plot(res; plotx=false, ploty=[output_idxs[3]], plotu=[3], plotxwithx̂=[3]))


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