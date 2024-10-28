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
using Statistics
# import RobustAndOptimalControl: named_ss

export step!, reset!, controlplot, start_processes!, stop_processes!
export ControlInterface


set_data_path(joinpath(@__DIR__, "..", "data"))
@with_kw mutable struct ControlInterface
    "symbolic inputs"
    inputs::Vector{Symbolics.Num}
    "symbolic outputs"
    outputs::Vector{Symbolics.Num}
    "initial inputs"
    u0::Vector{Float64}
    "initial state"
    x0::Vector{Float64}
    "wanted outputs"
    ry::Vector{Float64}
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
    output_idxs::Vector{Int}
    observed_idxs::Vector{Int}
    y_noise::Vector{Float64}
    error::Float64 = 0.0
    plotting::Bool = true
    linearized_channel::Channel{ModelPredictiveControl.LinMPC} = Channel{ModelPredictiveControl.LinMPC}(1)
    stepped_channel::Channel{Tuple{ModelPredictiveControl.LinMPC, 
                                ModelPredictiveControl.LinModel, 
                                Vector{Float64}, Vector{Float64}}} =
                    Channel{Tuple{ModelPredictiveControl.LinMPC, 
                                ModelPredictiveControl.LinModel, 
                                Vector{Float64}, Vector{Float64}}}(1)
end

include("mtk_interface.jl")

function ControlInterface(
        kite::KPS4_3L; 
        Ts::Float64=0.05, 
        x0::Vector{Float64}=kite.integrator.u,
        u0::Vector{Float64}=zeros(3), 
        ry::Union{Nothing, Vector{Float64}}=nothing, 
        noise::Float64=1e-3, 
        buffer_time::Int=20
        )
    # --- get symbolic inputs and outputs ---
    sys = kite.prob.f.sys
    inputs = [sys.set_values[i] for i in 1:3]
    outputs = vcat(
        vcat(sys.flap_angle), 
        reduce(vcat, collect(sys.pos[:, 4:kite.num_flap_C-1])), 
        reduce(vcat, collect(sys.pos[:, kite.num_flap_D+1:kite.num_A])),
        vcat(sys.tether_length),
        sys.heading_y,
        sys.depower,
        # sys.winch_force[3]
    )

    # --- generate ForwardDiff and MPC compatible f and h functions for linearization ---
    f!, h!, states, nu, nx, ny = generate_f_h(kite, inputs, outputs, QNDF(), Ts)
    nonlinmodel = NonLinModel(f!, h!, Ts, nu, nx, ny, solver=nothing)
    setname!(nonlinmodel, x=string.(states), u=string.(inputs), y=string.(outputs))

    linmodel, mpc, output_idxs, observed_idxs, optim, U_data, Y_data, Ry_data, X̂_data, y_noise, ry = 
        reset!(sys, nonlinmodel, x0, u0, ny, Ts, ry, noise, buffer_time)

    ci = ControlInterface(
        inputs = inputs,
        outputs = outputs,
        x0 = x0,
        u0 = u0,
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
        ry = ry,
        output_idxs = output_idxs,
        observed_idxs = observed_idxs,
        y_noise = y_noise,
    )
    return ci
end

function reset!(ci::ControlInterface; 
        x0::Vector{Float64}=ci.x0,
        u0::Vector{Float64}=ci.u0, 
        ry::Union{Nothing, Vector{Float64}}=nothing, 
        noise::Float64=1e-3,
        buffer_time::Int=20
        )
    ci.linmodel, ci.mpc, ci.output_idxs, ci.observed_idxs, ci.optim, ci.U_data, ci.Y_data, ci.Ry_data, ci.X̂_data, ci.y_noise, ci.ry =
        reset!(ci.kite.prob.f.sys, ci.nonlinmodel, x0, u0, ci.linmodel.ny, ci.Ts, ry, noise, buffer_time)
    return nothing
end
function reset!(sys, nonlinmodel, x0, u0, ny, Ts, ry, noise, buffer_time)
    # --- linearize model ---
    linmodel = linearize(nonlinmodel; x=x0, u=u0)

    # --- initialize outputs and plotting indexes ---
    if isnothing(ry)
        y0 = zeros(ny)
        nonlinmodel.h!(y0, x0, nothing, nothing)
        ry = y0
        ry[idx(linmodel.yname, sys.heading_y)] = deg2rad(0.0)
        ry[idx(linmodel.yname, sys.depower)] = 0.45
    end
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
    Mwt[idx(linmodel.yname, sys.heading_y)] = 1.0 / deg2rad(5.0)
    Mwt[idx(linmodel.yname, sys.depower)] = 1.0 / deg2rad(90)
    # Mwt[idx(linmodel.yname, sys.tether_length[1])] = 0.1
    # Mwt[idx(linmodel.yname, sys.tether_length[2])] = 0.1
    Mwt[idx(linmodel.yname, sys.tether_length[3])] = 1.0 / 10.0
    Nwt = fill(0.0, linmodel.nu)
    Lwt = fill(0.1, linmodel.nu)

    σR = fill(1e-4, linmodel.ny)
    σQ = fill(1e2, linmodel.nx)
    σQint_u=fill(1, linmodel.nu)
    nint_u=fill(1, linmodel.nu)
    estim = ModelPredictiveControl.UnscentedKalmanFilter(linmodel; nint_u, σQint_u, σQ, σR)

    Hp, Hc = 100, 2
    optim = JuMP.Model(HiGHS.Optimizer)
    mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt=Inf)

    umin, umax = [-20, -20, -500], [0.0, 0.0, 0.0]
    # max = 0.5
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
    # initstate!(mpc, zeros(3), y0) # TODO: check if needed

    # --- init data ---
    N = Int(round(buffer_time / Ts)) # buffer time is the amount of time to save
    U_data = fill(NaN, linmodel.nu, N)
    Y_data = fill(NaN, linmodel.ny, N)
    Ry_data = fill(NaN, linmodel.ny, N)
    X̂_data = fill(NaN, linmodel.nx + linmodel.ny, N)
    y_noise = fill(noise, linmodel.ny)
    return linmodel, mpc, output_idxs, observed_idxs, optim, U_data, Y_data, Ry_data, X̂_data, y_noise, ry
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

function step!(ci::ControlInterface, y; ry=ci.ry)
    x̂ = preparestate!(ci.mpc, y .+ ci.y_noise .* randn(ci.linmodel.ny))
    u = moveinput!(ci.mpc, ry)
    pop_append!(ci.U_data, u)
    pop_append!(ci.Y_data, y)
    pop_append!(ci.Ry_data, ry)
    pop_append!(ci.X̂_data, x̂)
    updatestate!(ci.mpc, u, y)
    @assert ci.plotting = true
    return u
end

function pop_append!(A::Matrix, vars::Vector)
    A[:, 1] .= 0
    A .= A[1:size(A, 1), vcat(2:size(A, 2), 1)]
    A[:, end] .= vars
    nothing
end

function plot_process(ci::ControlInterface)
    while ci.plotting
        display(controlplot(ci))
    end
end
function start_processes!(ci::ControlInterface)
    ci.plotting = true
    plot_thread = Threads.@spawn plot_process(ci)
    return plot_thread
end
function stop_processes!(ci::ControlInterface) ci.plotting = false end

function controlplot(ci::ControlInterface)
    res = SimResult(ci.mpc, ci.U_data, ci.Y_data; ci.Ry_data, ci.X̂_data)
    return Plots.plot(res; plotx=false, plotxwithx̂=ci.observed_idxs, ploty=ci.output_idxs, plotu=true, size=(1200, 900))
end


function linearize_process(ci::ControlInterface)
    ModelPredictiveControl.linearize!(ci.linmodel, ci.nonlinmodel; x, u)
    linearize!(ci.linmodel)
    setmodel!(mpc, linmodel)
    xnext = zeros(length(x))
    ModelPredictiveControl.f!(xnext, linmodel, x, u, nothing, nothing)
    ci.error = (norm(xnext .- x) - norm(linmodel.fop .- x)) / norm(linmodel.fop .- x)
    # @show norm(linmodel.fop .- linmodel.xop)
    # @show norm((linmodel.A * linmodel.xop + linmodel.Bu * linmodel.uop) .- linmodel.xop)
end
# @setup_workload begin
#     @compile_workload begin
#         init_set_values=[-0.1, -0.1, -70.0]
#         kite_model = KPS4_3L(KCU(se("system_3l.yaml")))
#         init_sim!(kite_model; prn=true, torque_control=true, init_set_values)
#         ci = ControlInterface(kite_model; Ts=0.05, u0=init_set_values)
#         nothing
#     end
# end

end