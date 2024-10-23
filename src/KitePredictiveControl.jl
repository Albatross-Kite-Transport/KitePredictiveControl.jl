module KitePredictiveControl

using ModelPredictiveControl
using PrecompileTools: @setup_workload, @compile_workload 
using ModelingToolkit
using KiteModels, ControlSystems, Serialization, OrdinaryDiffEqCore, OrdinaryDiffEqBDF, 
        RuntimeGeneratedFunctions, LinearAlgebra, SymbolicIndexingInterface, Plots, Base.Threads
using JuMP, DAQP, MadNLP, HiGHS, SeeToDee, NonlinearSolve, ForwardDiff # solvers
using ControlSystemIdentification, ControlSystemsBase, Parameters
using SymbolicIndexingInterface: parameter_values, state_values
using SciMLStructures
using ModelingToolkit: variable_index as idx
import ModelingToolkit.SciMLBase: successful_retcode
using PreallocationTools
# import RobustAndOptimalControl: named_ss

export step!, controlplot, live_plot, stop_plot
export ControlInterface

set_data_path(joinpath(@__DIR__, "..", "data"))
@with_kw mutable struct ControlInterface
    "symbolic inputs"
    inputs::Vector{Symbolics.Num}
    "symbolic outputs"
    outputs::Vector{Symbolics.Num}
    "initial outputs"
    y0::Vector{Float64}
    "initial state"
    x0::Vector{Float64}
    "parameters of the kite system"
    p0::ModelingToolkit.MTKParameters
    "sampling time"
    Ts::Float64
    "realistic KiteModels.jl model"
    kite::KiteModels.KPS4_3L
    "nonlinear model of the kite"
    nonlinmodel::ModelPredictiveControl.NonLinModel
    "linearized model of the kite"
    linmodel::ModelPredictiveControl.LinModel
    mpc::ModelPredictiveControl.LinMPC
    optim::JuMP.Model
    U_data::Matrix{Float64}
    Y_data::Matrix{Float64}
    Ry_data::Matrix{Float64}
    X̂_data::Matrix{Float64}
    X_data::Matrix{Float64}
    wanted_outputs::Vector{Float64}
    output_idxs::Vector{Int}
    observed_idxs::Vector{Int}
    y_noise::Vector{Float64}
    plotting::Bool = true
end

include("mtk_interface.jl")

function ControlInterface(kite; Ts = 0.05, u0 = zeros(3))
    # --- get symbolic model ---
    sym_model, inputs = model!(kite, kite.pos, kite.vel)
    sym_model = complete(sym_model)
    outputs = vcat(
        vcat(sym_model.flap_angle), 
        reduce(vcat, collect(sym_model.pos[:, 4:kite.num_flap_C-1])), 
        reduce(vcat, collect(sym_model.pos[:, kite.num_flap_D+1:kite.num_A])),
        vcat(sym_model.tether_length),
        sym_model.heading_y,
        sym_model.depower,
        # sym_model.winch_force[3]
    )
    sys = kite.prob.f.sys

    # --- generate ForwardDiff and MPC compatible f and h functions for linearization ---
    f!, h!, nu, nx, ny = generate_f_h(kite, inputs, outputs, Ts)
    nonlinmodel = NonLinModel(f!, h!, Ts, nu, nx, ny, solver=nothing)
    setname!(nonlinmodel, x=string.(unknowns(sys)), u=string.(inputs), y=string.(outputs))

    # --- linearize model ---
    x0 = deepcopy(kite.integrator.u)
    p0 = deepcopy(kite.integrator.p)
    @time linmodel = ModelPredictiveControl.linearize(nonlinmodel, x=x0, u=u0)
    @time linmodel = ModelPredictiveControl.linearize(nonlinmodel, x=x0, u=u0) # TODO: use sparse

    # --- initialize outputs ---
    y0 = zeros(ny)
    h!(y0, x0, nothing, nothing)

    time = 20 # amount of time to be saved in data buffer
    N = Int(round(time / Ts))

    output_idxs = vcat(
        idx(linmodel.yname, sys.heading_y),
        idx(linmodel.yname, sys.depower),
        # idx(linmodel.yname, sys.tether_length[1]),
        # idx(linmodel.yname, sys.tether_length[2]),
        idx(linmodel.yname, sys.tether_length[3]),
    )
    observed_idxs = vcat(
        # idx(sys, sys.pos[2, kite.num_A]),
        idx(sys, sys.tether_length[1]),
        idx(sys, sys.tether_length[2]),
        idx(sys, sys.tether_length[3]),
        idx(sys, sys.flap_angle[1]),
        idx(sys, sys.flap_angle[2]),
        # linmodel.nx + linmodel.ny
    )

    Mwt = fill(0.0, linmodel.ny)
    Mwt[idx(linmodel.yname, sys.heading_y)] = 1.0
    Mwt[idx(linmodel.yname, sys.depower)] = 1.0
    # Mwt[idx(linmodel.yname, sys.tether_length[1])] = 0.1
    # Mwt[idx(linmodel.yname, sys.tether_length[2])] = 0.1
    Mwt[idx(linmodel.yname, sys.tether_length[3])] = 1.0
    Nwt = fill(0.0, linmodel.nu)
    Lwt = fill(0.1, linmodel.nu)

    σR = fill(1e-4, linmodel.ny)
    σQ = fill(10, linmodel.nx)
    σQint_u=fill(1, linmodel.nu)
    nint_u=fill(1, linmodel.nu)
    estim = ModelPredictiveControl.UnscentedKalmanFilter(linmodel; nint_u, σQint_u, σQ, σR)

    # Hp_time, Hc_time = 1.0, Ts
    # Hp, Hc = Int(round(Hp_time / Ts)), Int(round(Hc_time / Ts))
    Hp, Hc = 20, 2
    optim = JuMP.Model(HiGHS.Optimizer)
    mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt=Inf)

    umin, umax = [-20, -20, -500], [0.0, 0.0, 0.0]
    max = 0.5
    # Δumin, Δumax = [-max, -max, -max*10], [max, max, max*10]
    ymin = fill(-Inf, linmodel.ny)
    ymax = fill(Inf, linmodel.ny)
    # ymin[idx(linmodel.yname, sys.tether_length[1])] = 57.0
    # ymin[idx(linmodel.yname, sys.tether_length[2])] = 57.0
    # ymin[idx(linmodel.yname, sys.tether_length[3])] = x0[idx(sys, sys.tether_length[3])] - 1.0
    # ymax[idx(linmodel.yname, sys.tether_length[1])] = x0[idx(sys, sys.tether_length[1])] + 0.3
    # ymax[idx(linmodel.yname, sys.tether_length[2])] = x0[idx(sys, sys.tether_length[2])] + 0.3
    # ymax[idx(linmodel.yname, sys.tether_length[3])] = x0[idx(sys, sys.tether_length[3])] + 1.0
    # ymin[end] = -1000
    # ymax[end] = 1000
    setconstraint!(mpc; umin, umax, ymin, ymax)
    # initstate!(mpc, zeros(3), y0)

    # --- init data ---
    U_data = fill(NaN, linmodel.nu, N)
    Y_data = fill(NaN, linmodel.ny, N)
    Ry_data = fill(NaN, linmodel.ny, N)
    X̂_data = fill(NaN, linmodel.nx + linmodel.ny, N)
    X_data = fill(NaN, linmodel.nx, N)

    wanted_outputs = y0
    wanted_outputs[idx(linmodel.yname, sys.heading_y)] = deg2rad(0.0)
    wanted_outputs[idx(linmodel.yname, sys.depower)] = 0.48
    # wanted_outputs[idx(linmodel.yname, sys.tether_length[3])] = 
    y_noise = fill(1e-3, linmodel.ny)

    ci = ControlInterface(
        inputs = inputs,
        outputs = outputs,
        y0 = y0,
        x0 = x0,
        p0 = p0,
        linmodel = linmodel,
        kite = kite,
        nonlinmodel = nonlinmodel,
        mpc = mpc,
        optim = optim,
        Ts = Ts,
        U_data = U_data,
        Y_data = Y_data,
        Ry_data = Ry_data,
        X̂_data = X̂_data,
        X_data = X_data,
        wanted_outputs = wanted_outputs,
        output_idxs = output_idxs,
        observed_idxs = observed_idxs,
        y_noise = y_noise
    )
    return ci
end

function reset(ci::ControlInterface, x0)
    ci.U_data .= NaN
    ci.Y_data .= NaN
    ci.Ry_data .= NaN
    ci.X̂_data .= NaN
    ci.X_data .= NaN
    ModelPredictiveControl.linearize!(ci.linmodel, ci.nonlinmodel)
end

function lin_ulin_sim(ci::ControlInterface)
    println("linear sanity check")
    u = [-0, -50, -70]
    res = sim!(ci.linmodel, 10, u; x0 = ci.x0)
    p1 = plot(res; plotx=false, ploty=ci.output_idxs, plotu=false, size=(900, 900))
    display(p1)
    # println("nonlinear sanity check")
    # res = sim!(plant, 10, u; x0 = ci.x0)
    # p2 = plot(res; plotx=false, ploty=false, plotu=false)
    # savefig(plot(p1, p2, layout=(1, 2)), "zeros.png")
    # @assert false
end

function step!(ci::ControlInterface, y; ry=ci.wanted_outputs)
    x̂ = preparestate!(ci.mpc, y)
    u = moveinput!(ci.mpc, ry)
    ModelPredictiveControl.linearize!(ci.linmodel, ci.nonlinmodel; x=x̂[1:ci.linmodel.nx], u)
    setmodel!(ci.mpc, ci.linmodel)
    pop_append!(ci.U_data, u)
    pop_append!(ci.Y_data, y)
    pop_append!(ci.Ry_data, ry)
    pop_append!(ci.X̂_data, x̂)
    # pop_append!(ci.X_data, x)
    updatestate!(ci.mpc, u, y)
    # updatestate!(ci.nonlinmodel, u)
    return u
end

function pop_append!(A::Matrix, vars::Vector)
    A[:, 1] .= 0
    A .= A[1:size(A, 1), vcat(2:size(A, 2), 1)]
    A[:, end] .= vars
    nothing
end

function plot_continuously(ci::ControlInterface)
    while ci.plotting
        display(controlplot(ci))
    end
end
function live_plot(ci::ControlInterface)
    ci.plotting = true
    plot_thread = Threads.@spawn plot_continuously(ci)
    return plot_thread
end
function stop_plot(ci::ControlInterface) ci.plotting = false end

function controlplot(ci::ControlInterface)
    res = SimResult(ci.mpc, ci.U_data, ci.Y_data; ci.Ry_data, ci.X̂_data, ci.X_data)
    return Plots.plot(res; plotx=false, plotxwithx̂=ci.observed_idxs, ploty=ci.output_idxs, plotu=true, size=(1200, 900))
end


function live_linearize(ci::ControlInterface)
    # TODO: threads spawn linearize on new x
end

# function init!(ci::ControlInterface)
#     # (; A, B, C, D) = linearize(ci.sys, ci.lin_fun, ci.x0, ci.p0; t=1.0);
#     # css = ss(A, B, C, D)
#     # dss = c2d(css, Ts, :zoh)
#     # linmodel = LinModel(dss.A, dss.B, dss.C, dss.B[:, end+1:end], dss.D[:, end+1:end], Ts)
#     # setstate!(linmodel, x0)
#     # setname!(linmodel; u=string.(inputs), y=string.(outputs), x=string.(unknowns(sys)))
#     # linmodel = create_lin_model(ci)

@setup_workload begin
    @compile_workload begin
        init_set_values=[-0.1, -0.1, -70.0]
        kite_model = KPS4_3L(KCU(se("system_3l.yaml")))
        init_sim!(kite_model; prn=true, torque_control=true, init_set_values)
        ci = ControlInterface(kite_model; Ts=0.05, u0=init_set_values)
        nothing
    end
end

    

end