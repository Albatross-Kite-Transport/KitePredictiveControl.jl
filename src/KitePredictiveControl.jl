# module KitePredictiveControl

using ModelPredictiveControl
using ModelingToolkit
using KiteModels, ControlSystems, Plots, Serialization, OrdinaryDiffEq, RuntimeGeneratedFunctions, LinearAlgebra, SymbolicIndexingInterface
using JuMP, DAQP, MadNLP, HiGHS, SeeToDee, NonlinearSolve, ForwardDiff # solvers
using ControlSystemIdentification, ControlSystemsBase
using ModelingToolkit: variable_index as idx
# import RobustAndOptimalControl: named_ss

# export run_controller

include("mtk_interface.jl")

optim = JuMP.Model(DAQP.Optimizer)

set_data_path(joinpath(pwd(), "data"))
if ! @isdefined kite
    kite::KPS4_3L = KPS4_3L(KCU(se("system_3l.yaml")))
    kite.torque_control = true
    KiteModels.init_sim!(kite; prn=true, torque_control=true)

    kite_model, inputs = model!(kite, kite.pos, kite.vel)
    kite_model = complete(kite_model)
    outputs = vcat(
        vcat(kite_model.flap_angle), 
        reduce(vcat, collect(kite_model.pos[:, 1:kite.num_flap_C-1])), 
        reduce(vcat, collect(kite_model.pos[:, kite.num_flap_D+1:kite.num_A])),
        vcat(kite_model.tether_length),
        # kite_model.winch_force[3]
    )
    initial_outputs = vcat(
        vcat(kite.flap_angle), 
        reduce(vcat, kite.pos[1:kite.num_flap_C-1]), 
        reduce(vcat, kite.pos[kite.num_flap_D+1:kite.num_A]),
        vcat(kite.tether_lengths),
        # norm.(kite.winch_forces)[3]
    )
    
    lin_fun, sys = ModelingToolkit.linearization_function(kite_model, inputs, outputs)
    [defaults(sys)[sys.pos[j, i]] = kite.pos[i][j] for j in 1:3 for i in 1:kite.num_flap_C-1]
    [defaults(sys)[sys.vel[j, i]] = kite.vel[i][j] for j in 1:3 for i in 1:kite.num_flap_C-1]
    [defaults(sys)[sys.pos[j, i]] = kite.pos[i][j] for j in 1:3 for i in kite.num_flap_D+1:kite.num_A]
    [defaults(sys)[sys.vel[j, i]] = kite.vel[i][j] for j in 1:3 for i in kite.num_flap_D+1:kite.num_A]
    [defaults(sys)[sys.flap_angle[i]] = kite.flap_angle[i] for i in 1:2]
    [defaults(sys)[sys.flap_vel[i]] = kite.flap_angle[i] for i in 1:2]
    [defaults(sys)[sys.tether_length[i]] = kite.tether_lengths[i] for i in 1:3]
    [defaults(sys)[sys.tether_vel[i]] = kite.tether_lengths[i] for i in 1:3]
end

Ts = 0.01
N = 1000
solver = QNDF(autodiff=false)
kite.integrator = OrdinaryDiffEq.init(kite.prob, solver; dt=Ts, abstol=kite.set.abs_tol, reltol=kite.set.rel_tol, save_on=false)

x_0 = copy(kite.integrator.u)
p_0 = copy(kite.integrator.p)
(; A, B, C, D) = linearize(sys, lin_fun, x_0, p_0; t=1.0);
css = ss(A, B, C, D)
dss = c2d(css, Ts)
model = LinModel(dss.A, dss.B, dss.C, dss.B[:, end+1:end], dss.D[:, end+1:end], Ts)
setstate!(model, x_0)
setname!(model; u=string.(inputs), y=string.(outputs), x=string.(unknowns(sys)))

f!, h! = generate_f_h(kite, inputs, outputs, Ts)
plant = NonLinModel(f!, h!, Ts, model.nu, model.nx, model.ny, solver=nothing)
setstate!(plant, x_0)

# println("linear sanity check")
# u = [-10, -0, -100]
# res = sim!(model, 10, u; x_0 = x_0)
# p1 = plot(res; plotx=vcat(idx(sys, sys.tether_length), idx(sys, sys.pos[2, kite.num_A])), ploty=false, plotu=false)

# println("nonlinear sanity check")
# res = sim!(plant, 10, u; x_0 = x_0)
# p2 = plot(res; plotx=vcat(idx(sys, sys.tether_length), idx(sys, sys.pos[2, kite.num_A])), ploty=false, plotu=false)
# savefig(plot(p1, p2, layout=(1, 2)), "zeros.png")
 
# @assert false

println("linear mpc")
output_idxs = vcat(
    idx(model.yname, sys.pos[2, kite.num_A]),
    # idx(model.yname, sys.winch_force[3]),
    idx(model.yname, sys.tether_length[3])
)
observed_idxs = vcat(
    # idx(sys, sys.pos[2, kite.num_A]),
    idx(sys, sys.tether_length[3]),
    idx(sys, sys.tether_length[2]),
    idx(sys, sys.tether_length[1]),
    # model.nx + model.ny
)

Mwt = fill(0.0, model.ny)
Mwt[idx(model.yname, sys.pos[2, kite.num_A])] = 10.0
# Mwt[idx(model.yname, sys.tether_length[3])] = 0.1
Nwt = fill(10, model.nu)

σR = fill(0.01, model.ny)
σQ = fill(1000/model.nx, model.nx)
σQint_u=fill(1, model.nu)
nint_u=fill(1, model.nu)
estim = ModelPredictiveControl.KalmanFilter(model; nint_u, σQint_u, σQ, σR)

Hp, Hc = 40, 10
mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=1e9, optim)

umin, umax = [-10, -10, -100], [0, 0, 0]
ymin = fill(-Inf, model.ny)
ymax = fill(Inf, model.ny)
ymin[idx(model.yname, sys.tether_length[3])] = 0.0
ymax[idx(model.yname, sys.tether_length[3])] = 51.0
# ymin[end] = -1000
# ymax[end] = 1000
setconstraint!(mpc; umin, umax, ymin, ymax)

println("stepping linear mpc with nonlinear plant")

# estim = ModelPredictiveControl.KalmanFilter(model; σQ, σR, nint_u, σQint_u)
# mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=1e5)
# mpc = setconstraint!(mpc; umin, umax)
function sim_adapt!(mpc, sys, model, N, ry, plant, x_0, y_step=zeros(plant.ny))
    U_data, Y_data, Ry_data, X̂_data, X_data = zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N), zeros(plant.nx+plant.ny, N), zeros(plant.nx, N)
    setstate!(plant, x_0)
    initstate!(mpc, zeros(3), plant())
    # setstate!(mpc, x_0)
    for i = 1:N
        @show i
        y = plant() + y_step
        x̂ = preparestate!(mpc, y)
        u = moveinput!(mpc, ry)
        (; A, B, C, D) = linearize(sys, lin_fun, plant.x0, p_0; t=1.0);
        css = ss(A, B, C, D)
        dss = c2d(css, Ts)
        model.A .= dss.A
        model.Bu .= dss.B
        model.C .= dss.C
        # setmodel!(mpc, model)
        U_data[:,i], Y_data[:,i], Ry_data[:,i], X̂_data[:, i], X_data[:, i] = u, y, ry, x̂, plant.x0
        updatestate!(mpc, u, y) # update mpc state estimate
        updatestate!(plant, u)  # update plant simulator
    end
    res = SimResult(mpc, U_data, Y_data; Ry_data, X̂_data, X_data)
    return res
end
ry = initial_outputs .+ 0.02
res = sim_adapt!(mpc, sys, model, N, ry, plant, x_0)
display(plot(res; plotx=false, ploty=output_idxs, plotu=true, plotxwithx̂=observed_idxs))

nothing

# end