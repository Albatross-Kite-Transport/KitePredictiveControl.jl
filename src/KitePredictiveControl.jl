module KitePredictiveControl

using ModelPredictiveControl
using PrecompileTools: @setup_workload, @compile_workload
using KiteModels, ControlSystems, Serialization, OrdinaryDiffEq,
    LinearAlgebra, Plots, Base.Threads
using JuMP, HiGHS # solvers
using Parameters
using SymbolicIndexingInterface: parameter_values, state_values
using ModelingToolkit: variable_index as idx, unknowns
using ModelingToolkit.SciMLBase: successful_retcode
using SymbolicIndexingInterface: getu, setp
import Symbolics: Num

export ControlInterface
export step!, reset!, controlplot, start_processes!, stop_processes!

set_data_path(joinpath(@__DIR__, "..", "data"))
@with_kw mutable struct ControlInterface
    "symbolic inputs"
    inputs::Vector{Num}
    "symbolic outputs"
    outputs::Vector{Num}
    "initial inputs"
    u0::Vector{Float64}
    "initial state"
    x0::Vector{Float64}
    "initial simple state"
    x_simple_0::Vector{Float64}
    "wanted outputs"
    ry::Vector{Float64}
    "sampling time"
    Ts::Float64
    "nonlinear KiteModels.jl model"
    kite::KiteModels.KPS4_3L
    "complex x0 to simple x_plus nonlinear function"
    f!::Function
    "observer function"
    h!::Function
    "linearized model of the kite"
    linmodel::ModelPredictiveControl.LinModel
    mpc::ModelPredictiveControl.LinMPC
    optim::JuMP.Model
    AB::Matrix{Float64}
    AB_pattern::Matrix{Float64}
    U_data::Matrix{Float64}
    Y_data::Matrix{Float64}
    Ry_data::Matrix{Float64}
    X̂_data::Matrix{Float64}
    output_idxs::Vector{Int}
    observed_idxs::Vector{Int}
    y_noise::Vector{Float64}
    error::Float64 = 0.0
    "how many time steps to look into the future when linearizing"
    time_multiplier::Int = 40
    plotting::Bool = true
    # linearized_channel::Channel{ModelPredictiveControl.LinMPC} = Channel{ModelPredictiveControl.LinMPC}(1)
    # stepped_channel::Channel{Tuple{ModelPredictiveControl.LinMPC,
    #     ModelPredictiveControl.LinModel,
    #     Vector{Float64},Vector{Float64}}} =
    #     Channel{Tuple{ModelPredictiveControl.LinMPC,
    #         ModelPredictiveControl.LinModel,
    #         Vector{Float64},Vector{Float64}}}(1)
end


function ControlInterface(
    kite::KPS4_3L;
    Ts::Float64=0.05,
    x0::Vector{Float64}=kite.integrator.u,
    u0::Vector{Float64}=zeros(3),
    ry::Union{Nothing,Vector{Float64}}=nothing,
    noise::Float64=1e-3,
    buffer_time::Int=20,
    time_multiplier::Int=10,
)
    x0 = copy(x0)
    # --- get symbolic inputs and outputs ---
    sys = kite.prob.f.sys
    inputs = [sys.set_values[i] for i in 1:3]
    outputs = vcat(
        sys.heading_y,
        sys.turn_rate_y,
        sys.depower,
        sys.depower_vel,
        vcat(sys.tether_length),
        vcat(sys.tether_vel),
    )

    # --- generate ForwardDiff and MPC compatible f and h functions for linearization ---
    f!, h!, simple_state, nu, nx, nsx = generate_f_h(kite, inputs, outputs, QNDF(), Ts)
    # nonlinmodel = NonLinModel(f!, h!, Ts, nu, nx, ny, solver=nothing)
    # setname!(nonlinmodel, x=string.(states), u=string.(inputs), y=string.(outputs))

    linmodel, mpc, output_idxs, observed_idxs, optim, U_data, Y_data, Ry_data, X̂_data, y_noise, x_simple_0, ry, AB, AB_pattern =
        reset!(sys, simple_state, inputs, time_multiplier, x0, u0, nsx, nu, Ts, ry, noise, buffer_time, f!, h!)

    ci = ControlInterface(
        inputs=inputs,
        outputs=outputs,
        x0=x0,
        x_simple_0=x_simple_0,
        u0=u0,
        linmodel=linmodel,
        kite=kite,
        (f!)=f!,
        (h!)=h!,
        mpc=mpc,
        optim=optim,
        Ts=Ts,
        U_data=U_data,
        Y_data=Y_data,
        Ry_data=Ry_data,
        X̂_data=X̂_data,
        ry=ry,
        output_idxs=output_idxs,
        observed_idxs=observed_idxs,
        y_noise=y_noise,
        AB=AB,
        AB_pattern=AB_pattern,
        time_multiplier=time_multiplier
    )
    return ci
end

include("mtk_interface.jl")
include("smartdiff.jl")

function reset!(ci::ControlInterface;
    x0::Vector{Float64}=ci.x0,
    u0::Vector{Float64}=ci.u0,
    ry::Union{Nothing,Vector{Float64}}=nothing,
    noise::Float64=1e-3,
    buffer_time::Int=20
)
    ci.linmodel, ci.mpc, ci.output_idxs, ci.observed_idxs, ci.optim, ci.U_data, ci.Y_data, ci.Ry_data, ci.X̂_data, ci.y_noise, ci.ry =
        reset!(ci.kite.prob.f.sys, x0, u0, ci.linmodel.ny, ci.Ts, ry, noise, buffer_time)
    return nothing
end
function reset!(sys, simple_state, inputs, time_multiplier, x0, u0, nsx, nu, Ts, ry, noise, buffer_time, f!, h!)
    # function yidx(name::Vector{String}, var)
    #     return findfirst(x -> x == string(var), name)
    # end
    yidx = var -> findfirst(x -> x == string(var), linmodel.yname)
    xidx = var -> findfirst(x -> x == string(var), linmodel.xname)

    # --- linearize model ---
    x_simple_0 = zeros(nsx)
    linmodel, AB, AB_pattern = linearize(sys, f!, h!, nsx, nu,
        string.(simple_state), string.(inputs),
        x0, x_simple_0, u0, Ts; time_multiplier)

    # --- initialize outputs and plotting indexes ---
    if isnothing(ry)
        y0 = zeros(nsx)
        h!(y0, x0)
        ry = y0
        ry[yidx(sys.heading_y)] = deg2rad(0.0)
        ry[yidx(sys.depower)] = 0.45
    end
    output_idxs = vcat(
        yidx(sys.heading_y),
        yidx(sys.depower),
        # yidx(sys.tether_length[1]),
        # yidx(sys.tether_length[2]),
        yidx(sys.tether_length[3]),
    )
    observed_idxs = vcat(
        # xidx(sys.pos[2, kite.num_A]),
        xidx(sys.tether_length[1]),
        xidx(sys.tether_length[2]),
        xidx(sys.tether_length[3]),
        xidx(sys.depower),
        xidx(sys.heading_y),
        # linmodel.nx + linmodel.ny
    )

    Mwt = fill(0.0, linmodel.ny)
    Mwt[yidx(sys.heading_y)] = 10.0
    Mwt[yidx(sys.depower)] = 1.0
    # Mwt[yidx(sys.tether_length[1])] = 0.1
    # Mwt[yidx(sys.tether_length[2])] = 0.1
    Mwt[yidx(sys.tether_length[3])] = 0.1
    Nwt = fill(0.0, linmodel.nu)
    Lwt = fill(0.1, linmodel.nu)

    σR = fill(1e-4, linmodel.ny)
    σQ = fill(1e2, linmodel.nx)
    σQint_u = fill(1, linmodel.nu)
    nint_u = fill(1, linmodel.nu)
    estim = ModelPredictiveControl.UnscentedKalmanFilter(linmodel; nint_u, σQint_u, σQ, σR)

    Hp, Hc = 40, 1
    optim = JuMP.Model(HiGHS.Optimizer)
    mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt=Inf)

    umin, umax = [-20, -20, -500], [0.0, 0.0, 0.0] # TODO: torque control with u0 = -winch_forces * drum_radius
    # max = 0.5
    # Δumin, Δumax = [-max, -max, -max*10], [max, max, max*10]
    ymin = fill(-Inf, linmodel.ny)
    ymax = fill(Inf, linmodel.ny)
    ymax[yidx(sys.depower)] = 0.6
    setconstraint!(mpc; umin, umax, ymin, ymax)
    # initstate!(mpc, zeros(3), y0) # TODO: check if needed

    # --- init data ---
    N = Int(round(buffer_time / Ts)) # buffer time is the amount of time to save
    U_data = fill(NaN, linmodel.nu, N)
    Y_data = fill(NaN, linmodel.ny, N)
    Ry_data = fill(NaN, linmodel.ny, N)
    X̂_data = fill(NaN, estim.nx̂, N)
    y_noise = fill(noise, linmodel.ny)
    return linmodel, mpc, output_idxs, observed_idxs, optim, U_data, Y_data, Ry_data, X̂_data, y_noise, x_simple_0, ry, AB, AB_pattern
end


function lin_ulin_sim(ci::ControlInterface)
    println("linear sanity check")
    u = [-0, -50, -70]
    res = sim!(ci.linmodel, 10, u; x0=ci.x0)
    p1 = plot(res; plotx=false, ploty=ci.output_idxs, plotu=false, size=(900, 900))
    display(p1)
    # println("nonlinear sanity check")
    # res = sim!(plant, 10, u; x0 = ci.x0)
    # p2 = plot(res; plotx=false, ploty=false, plotu=false)
    # savefig(plot(p1, p2, layout=(1, 2)), "zeros.png")
    # @assert false
end

function step!(ci::ControlInterface, x, y; ry=ci.ry, rheading=nothing)
    if !isnothing(rheading)
        ci.ry[1] = rheading
    end
    x̂ = preparestate!(ci.mpc, y .+ ci.y_noise .* randn(ci.linmodel.ny))
    u = moveinput!(ci.mpc, ry)
    linearize!(ci, ci.linmodel, x, u)
    display(linearization_plot(ci, x, u))
    @show ci.linmodel.A[1, 2]
    setmodel!(ci.mpc, ci.linmodel)
    pop_append!(ci.U_data, u)
    pop_append!(ci.Y_data, y)
    pop_append!(ci.Ry_data, ry)
    pop_append!(ci.X̂_data, x̂)
    updatestate!(ci.mpc, u, y)
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
        # display(controlplot(ci))
    end
end
function start_processes!(ci::ControlInterface)
    ci.plotting = true
    plot_thread = Threads.@spawn plot_process(ci)
    return plot_thread
end
function stop_processes!(ci::ControlInterface)
    ci.plotting = false
end

function controlplot(ci::ControlInterface)
    res = SimResult(ci.mpc, ci.U_data, ci.Y_data; ci.Ry_data, ci.X̂_data)
    return Plots.plot(res; plotx=false, plotxwithx̂=ci.observed_idxs, ploty=ci.output_idxs, plotu=true, size=(1200, 900))
end

"""
Plot the linearization projected n timesteps into the future
"""
function linearization_plot(ci::ControlInterface, x0, u0; n::Int=5)
    linmodel = deepcopy(ci.linmodel)
    u = u0 .+ [0.0, 1.0, 0.0]
    x_simple_0 = copy(linmodel.xop)
    ci.h!(x_simple_0, x0)

    x_simple_plus = similar(x_simple_0)
    ci.f!(x_simple_plus, x0, u, ci.Ts*n)
    lin_plus = linmodel.A * x_simple_0 + linmodel.Bu * u
    for _ in 1:n-1
        lin_plus = linmodel.A * lin_plus + linmodel.Bu * u
    end
    p = plot()
    idx = 1
    plot!(p, [x_simple_0[idx], x_simple_plus[idx]], label="nonlin")
    plot!(p, [x_simple_0[idx], lin_plus[idx]], label="lin", ylim=(-0.01, 0.01))
    return p
end

function linearize_process(ci::ControlInterface)
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