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

include("mtk_interface.jl")

set_data_path(joinpath(@__DIR__, "..", "data"))
@with_kw mutable struct ControlInterface
    lin_fun::Function
    sys::ODESystem
    inputs::Vector{Symbolics.Num}
    outputs::Vector{Symbolics.Num}
    initial_outputs::Vector{Float64}
    x_0::Vector{Float64}
    p_0::ModelingToolkit.MTKParameters
    Ts::Float64
    model::ModelPredictiveControl.LinModel
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

function ControlInterface(kite; Ts = 0.05)
    KiteModels.init_sim!(kite; prn=true, torque_control=kite.torque_control)
    kite_model, inputs = model!(kite, kite.pos, kite.vel)
    kite_model = complete(kite_model)
    outputs = vcat(
        vcat(kite_model.flap_angle), 
        reduce(vcat, collect(kite_model.pos[:, 4:kite.num_flap_C-1])), 
        reduce(vcat, collect(kite_model.pos[:, kite.num_flap_D+1:kite.num_A])),
        vcat(kite_model.tether_length),
        kite_model.heading
        # kite_model.winch_force[3]
    )
    get_y = getu(kite.integrator.sol, outputs)
    initial_outputs = kite.integrator[outputs]
    lin_fun, sys = ModelingToolkit.linearization_function(kite_model, inputs, outputs)

    time = 5 # amount of time to be saved
    N = Int(round(time / Ts))
    # solver = QNDF(autodiff=false)
    # kite.integrator = OrdinaryDiffEqCore.init(kite.prob, solver; dt=Ts, abstol=kite.set.abs_tol, reltol=kite.set.rel_tol, save_on=false)
    init_sim!(kite; torque_control=kite.torque_control)

    x_0 = deepcopy(kite.integrator.u)
    p_0 = deepcopy(kite.integrator.p)

    model = create_lin_model(sys, lin_fun, Ts, inputs, outputs, x_0, p_0)
    # f!, h! = generate_f_h(kite, inputs, outputs, Ts)
    # plant = NonLinModel(f!, h!, Ts, model.nu, model.nx, model.ny, solver=nothing)
    # setstate!(plant, x_0)
    output_idxs = vcat(
        idx(model.yname, sys.heading),
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
    Mwt[idx(model.yname, sys.heading)] = 0.0
    Mwt[idx(model.yname, sys.tether_length[3])] = 1.0
    Nwt = fill(0.1, model.nu)

    σR = fill(0.01, model.ny)
    σQ = fill(1000/model.nx, model.nx)
    σQint_u=fill(1, model.nu)
    nint_u=fill(1, model.nu)
    estim = ModelPredictiveControl.KalmanFilter(model; nint_u, σQint_u, σQ, σR)

    Hp_time, Hc_time = 1.0, 0.2
    Hp, Hc = Int(round(Hp_time / Ts)), Int(round(Hc_time / Ts))
    optim = JuMP.Model(DAQP.Optimizer)
    mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=1e9, optim)

    umin, umax = [-1, -1, -1], [1, 1, 1]
    # Δumin, Δumax = [-0.1, -0.1, -0.], [0.1, 0.1, 0.1]
    ymin = fill(-Inf, model.ny)
    ymax = fill(Inf, model.ny)
    ymin[idx(model.yname, sys.tether_length[1])] = x_0[idx(sys, sys.tether_length[1])] - 0.1
    ymin[idx(model.yname, sys.tether_length[2])] = x_0[idx(sys, sys.tether_length[2])] - 0.1
    ymin[idx(model.yname, sys.tether_length[3])] = x_0[idx(sys, sys.tether_length[3])] - 1.0
    ymax[idx(model.yname, sys.tether_length[1])] = x_0[idx(sys, sys.tether_length[1])] + 1.0
    ymax[idx(model.yname, sys.tether_length[2])] = x_0[idx(sys, sys.tether_length[2])] + 1.0
    ymax[idx(model.yname, sys.tether_length[3])] = x_0[idx(sys, sys.tether_length[3])] + 1.0
    # ymin[end] = -1000
    # ymax[end] = 1000
    setconstraint!(mpc; umin, umax, ymin, ymax)

    U_data, Y_data, Ry_data, X̂_data, X_data = 
        fill(NaN, model.nu, N), fill(NaN, model.ny, N), fill(NaN, model.ny, N), fill(NaN, model.nx+model.ny, N), fill(NaN, model.nx, N)
    wanted_outputs = initial_outputs .+ 0.02
    y_noise = fill(1e-3, model.ny)

    ci = ControlInterface(
        lin_fun = lin_fun,
        sys = sys,
        inputs = inputs,
        outputs = outputs,
        initial_outputs = initial_outputs,
        x_0 = x_0,
        p_0 = p_0,
        model = model,
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

# function lin_ulin_sim(ci::ControlInterface)
#     println("linear sanity check")
#     u = [-1, -1, -1]
#     res = sim!(ci.model, 10, u; x_0 = x_0)
#     p1 = plot(res; plotx=vcat(idx(sys, sys.tether_length), idx(sys, sys.pos[2, kite.num_A])), ploty=false, plotu=false)
    
#     println("nonlinear sanity check")
#     res = sim!(plant, 10, u; x_0 = x_0)
#     p2 = plot(res; plotx=vcat(idx(sys, sys.tether_length), idx(sys, sys.pos[2, kite.num_A])), ploty=false, plotu=false)
#     savefig(plot(p1, p2, layout=(1, 2)), "zeros.png")
#     @assert false
# end


function create_lin_model(ci, x, model=nothing; kwargs...)
    return create_lin_model(ci.sys, ci.lin_fun, ci.Ts, ci.inputs, ci.outputs, x, ci.p_0, model; kwargs...)
end
function create_lin_model(sys, lin_fun, Ts, inputs, outputs, x, p, model=nothing; init_state=true, init_name=true, smooth=0.05)
    (; A, B, C, D) = linearize(sys, lin_fun, x, p; t=1.0);
    css = ss(A, B, C, D)
    dss = c2d(css, Ts, :zoh)
    if isnothing(model)
        model = LinModel(dss.A, dss.B, dss.C, dss.B[:, end+1:end], dss.D[:, end+1:end], Ts)
    else
        model = LinModel(
            dss.A*smooth + model.A*(1-smooth), 
            dss.B*smooth + model.Bu*(1-smooth), 
            dss.C*smooth + model.C*(1-smooth),
            model.Bd, model.Dd, model.Ts)
    end
    init_state && setstate!(model, x)
    init_name && setname!(model; u=string.(inputs), y=string.(outputs), x=string.(unknowns(sys)))
    return model
end

function step!(ci::ControlInterface, integrator; ry=ci.wanted_outputs)
    x = integrator.u
    y = ci.get_y(integrator) + ci.y_noise.*randn(ci.model.ny)
    x̂ = preparestate!(ci.mpc, y)
    u = moveinput!(ci.mpc, ry)
    ci.model = create_lin_model(ci, x, ci.model; init_name=false)
    setmodel!(ci.mpc, ci.model)
    # ci.U_data[:,i], ci.Y_data[:,i], ci.Ry_data[:,i], ci.X̂_data[:, i], ci.X_data[:, i] = u, y, ry, x̂, x
    pop_append!(ci.U_data, u)
    pop_append!(ci.Y_data, y)
    pop_append!(ci.Ry_data, ry)
    pop_append!(ci.X̂_data, x̂)
    pop_append!(ci.X_data, x)

    updatestate!(ci.mpc, u, y) # update mpc state estimate
    # updatestate!(ci.plant, u)  # update plant simulator
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
        sleep(1)
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
    return Plots.plot(res; plotx=false, ploty=ci.output_idxs, plotu=true)
end

# function init!(ci::ControlInterface)
#     # (; A, B, C, D) = linearize(ci.sys, ci.lin_fun, ci.x_0, ci.p_0; t=1.0);
#     # css = ss(A, B, C, D)
#     # dss = c2d(css, Ts, :zoh)
#     # model = LinModel(dss.A, dss.B, dss.C, dss.B[:, end+1:end], dss.D[:, end+1:end], Ts)
#     # setstate!(model, x_0)
#     # setname!(model; u=string.(inputs), y=string.(outputs), x=string.(unknowns(sys)))
#     # model = create_lin_model(ci)

    
# end

# function sim_adapt!(ci::ControlInterface, mpc, sys, model, N, ry, plant, x_0, p_0, y_step=zeros(plant.ny))
#     U_data, Y_data, Ry_data, X̂_data, X_data = zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N), zeros(plant.nx+plant.ny, N), zeros(plant.nx, N)
#     setstate!(plant, x_0)
#     initstate!(mpc, zeros(3), plant())
#     # setstate!(mpc, x_0)
#     step_time = 0.0
#     for i = 1:N
#         Core.println("time = ", i*Ts)
#         step_time += @elapsed begin
#             y = plant() + y_step + y_noise.*randn(plant.ny)
#             x̂ = preparestate!(mpc, y)
#             u = moveinput!(mpc, ry)

#             # (; A, B, C, D) = linearize(sys, lin_fun, x, p_0; t=1.0);
#             # css = ss(A, B, C, D)
#             # dss = c2d(css, Ts, :zoh)
#             # model = LinModel(
#             #     dss.A*0.05 + model.A*0.95, 
#             #     dss.B*0.05 + model.Bu*0.95, 
#             #     dss.C*0.05 + model.C*0.95, 
#             #     dss.B[:, end+1:end], dss.D[:, end+1:end], Ts)
#             model = create_lin_model(ci, plant.x_0, model)
#             setmodel!(mpc, model)

#             U_data[:,i], Y_data[:,i], Ry_data[:,i], X̂_data[:, i], X_data[:, i] = u, y, ry, x̂, x
#             updatestate!(mpc, u, y) # update mpc state estimate
#         end
#         updatestate!(plant, u)  # update plant simulator
#     end
#     res = SimResult(mpc, U_data, Y_data; Ry_data, X̂_data, X_data)
#     return res, step_time / N
# end

# """
# stepping linear mpc with nonlinear plant
# """
# function lin_mpc(ci::ControlInterface)
#     # estim = ModelPredictiveControl.KalmanFilter(model; σQ, σR, nint_u, σQint_u)
#     # mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=1e5)
#     # mpc = setconstraint!(mpc; umin, umax)
#     ry = initial_outputs .+ 0.02
#     res, step_time = sim_adapt!(ci, mpc, sys, model, N, ry, plant, x_0, p_0)
#     display(plot(res; plotx=false, ploty=output_idxs, plotu=true, plotxwithx̂=observed_idxs))
#     println("Times realtime: ", Ts / step_time)
# end

end