module KitePredictiveControl

using ModelPredictiveControl
using ModelingToolkit
using KiteModels, ControlSystems, Serialization, OrdinaryDiffEqCore, OrdinaryDiffEqBDF, 
        RuntimeGeneratedFunctions, LinearAlgebra, SymbolicIndexingInterface, Plots, Base.Threads
using JuMP, DAQP, MadNLP, HiGHS, SeeToDee, NonlinearSolve, ForwardDiff # solvers
using ControlSystemIdentification, ControlSystemsBase, Parameters
using ModelingToolkit: variable_index as idx
# import RobustAndOptimalControl: named_ss

export step!, controlplot, live_plot, stop_plot
export ControlInterface

set_data_path(joinpath(@__DIR__, "..", "data"))
@with_kw mutable struct ControlInterface
    "fast linearization function"
    lin_fun::Function
    "simplified system of the kite model"
    sys::ODESystem
    "symbolic inputs"
    inputs::Vector{Symbolics.Num}
    "symbolic outputs"
    outputs::Vector{Symbolics.Num}
    "initial outputs"
    y_0::Vector{Float64}
    "initial state"
    x_0::Vector{Float64}
    "parameters of the kite system"
    p_0::ModelingToolkit.MTKParameters
    "sampling time"
    Ts::Float64
    "realistic nonlinear model of the kite"
    kite::KPS4_3L
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
    get_y::Union{SymbolicIndexingInterface.MultipleGetters, SymbolicIndexingInterface.TimeDependentObservedFunction}
    plotting::Bool = true
end

include("mtk_interface.jl")

function ControlInterface(kite; Ts = 0.05, init_set_values = zeros(3))
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
    get_y = getu(kite.integrator.sol, outputs)
    y_0 = kite.integrator[outputs]
    lin_fun, sys = ModelingToolkit.linearization_function(sym_model, inputs, outputs)

    time = 20 # amount of time to be saved
    N = Int(round(time / Ts))
    # solver = QNDF(autodiff=false)
    # kite.integrator = OrdinaryDiffEqCore.init(kite.prob, solver; dt=Ts, abstol=kite.set.abs_tol, reltol=kite.set.rel_tol, save_on=false)
    # init_sim!(kite; prn=true, torque_control=kite.torque_control)

    x_0 = deepcopy(kite.integrator.u)
    p_0 = deepcopy(kite.integrator.p)

    linmodel = linearize(kite, sys, lin_fun, get_y, x_0, init_set_values, p_0, Ts)
    setname!(linmodel, x=string.(unknowns(sys)), u=string.(inputs), y=string.(outputs))

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
    Hp, Hc = 10, 2
    optim = JuMP.Model(HiGHS.Optimizer)
    mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt=Inf)

    umin, umax = [-20, -20, -500], [0.0, 0.0, 0.0]
    max = 0.5
    # Δumin, Δumax = [-max, -max, -max*10], [max, max, max*10]
    ymin = fill(-Inf, linmodel.ny)
    ymax = fill(Inf, linmodel.ny)
    # ymin[idx(linmodel.yname, sys.tether_length[1])] = 57.0
    # ymin[idx(linmodel.yname, sys.tether_length[2])] = 57.0
    # ymin[idx(linmodel.yname, sys.tether_length[3])] = x_0[idx(sys, sys.tether_length[3])] - 1.0
    # ymax[idx(linmodel.yname, sys.tether_length[1])] = x_0[idx(sys, sys.tether_length[1])] + 0.3
    # ymax[idx(linmodel.yname, sys.tether_length[2])] = x_0[idx(sys, sys.tether_length[2])] + 0.3
    # ymax[idx(linmodel.yname, sys.tether_length[3])] = x_0[idx(sys, sys.tether_length[3])] + 1.0
    # ymin[end] = -1000
    # ymax[end] = 1000
    setconstraint!(mpc; umin, umax, ymin, ymax)
    # initstate!(mpc, zeros(3), y_0)

    U_data, Y_data, Ry_data, X̂_data, X_data = 
        fill(NaN, linmodel.nu, N), fill(NaN, linmodel.ny, N), fill(NaN, linmodel.ny, N), fill(NaN, linmodel.nx+linmodel.ny, N), fill(NaN, linmodel.nx, N)
    wanted_outputs = y_0
    wanted_outputs[idx(linmodel.yname, sys.heading_y)] = deg2rad(0.0)
    wanted_outputs[idx(linmodel.yname, sys.depower)] = 0.48
    # wanted_outputs[idx(linmodel.yname, sys.tether_length[3])] = 
    y_noise = fill(0.01, linmodel.ny)

    ci = ControlInterface(
        lin_fun = lin_fun,
        sys = sys,
        inputs = inputs,
        outputs = outputs,
        y_0 = y_0,
        x_0 = x_0,
        p_0 = p_0,
        linmodel = linmodel,
        kite = kite,
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
        y_noise = y_noise,
        get_y = get_y
    )
    return ci
end

function reset(ci::ControlInterface)

end

function lin_ulin_sim(ci::ControlInterface)
    println("linear sanity check")
    u = [-0, -50, -70]
    res = sim!(ci.linmodel, 10, u; x_0 = ci.x_0)
    p1 = plot(res; plotx=ci.observed_idxs, ploty=ci.output_idxs, plotu=false, size=(900, 900))
    display(p1)
    # println("nonlinear sanity check")
    # res = sim!(plant, 10, u; x_0 = ci.x_0)
    # p2 = plot(res; plotx=ci.observed_idxs, ploty=false, plotu=false)
    # savefig(plot(p1, p2, layout=(1, 2)), "zeros.png")
    # @assert false
end



function step!(ci::ControlInterface, integrator; ry=ci.wanted_outputs)
    x = integrator.u
    y = ci.get_y(integrator) .+ ci.y_noise.*randn(ci.linmodel.ny)
    x̂ = preparestate!(ci.mpc, y)
    u = moveinput!(ci.mpc, ry)
    # ci.linmodel = create_lin_model(ci, x, y, ci.linmodel; init_name=true, smooth=0.0)
    linearize!(ci, ci.linmodel, x, u, ci.p_0)
    setmodel!(ci.mpc, ci.linmodel)
    pop_append!(ci.U_data, u)
    pop_append!(ci.Y_data, y)
    pop_append!(ci.Ry_data, ry)
    pop_append!(ci.X̂_data, x̂)
    pop_append!(ci.X_data, x)
    updatestate!(ci.mpc, u, y) # update mpc state estimate
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
#     # (; A, B, C, D) = linearize(ci.sys, ci.lin_fun, ci.x_0, ci.p_0; t=1.0);
#     # css = ss(A, B, C, D)
#     # dss = c2d(css, Ts, :zoh)
#     # linmodel = LinModel(dss.A, dss.B, dss.C, dss.B[:, end+1:end], dss.D[:, end+1:end], Ts)
#     # setstate!(linmodel, x_0)
#     # setname!(linmodel; u=string.(inputs), y=string.(outputs), x=string.(unknowns(sys)))
#     # linmodel = create_lin_model(ci)

    

end