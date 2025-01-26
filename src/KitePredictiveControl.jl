module KitePredictiveControl

using ModelPredictiveControl
using PrecompileTools: @setup_workload, @compile_workload
using KiteModels
using ControlSystems, Serialization, OrdinaryDiffEq,
    LinearAlgebra, Plots, Base.Threads, PreallocationTools
using JuMP, HiGHS # solvers
using SymbolicIndexingInterface: parameter_values, state_values
using ModelingToolkit: variable_index as idx, unknowns
using ModelingToolkit.SciMLBase: successful_retcode
using SymbolicIndexingInterface: getu, setp
import Symbolics: Num

export ControlInterface
export step!, controlplot, start_processes!, stop_processes!

set_data_path(joinpath(@__DIR__, "..", "data"))
mutable struct ControlInterface
    "symbolic input"
    input::Vector{Num}
    "symbolic output"
    output::Vector{Num}
    "initial input"
    u0::Vector{Float64}
    "initial state"
    x0::Vector{Float64}
    "wanted output"
    ry::Vector{Float64}
    "sampling time [seconds]"
    Ts::Float64
    "predict horizon [number of steps]"
    Hp::Int
    "nonlinear KiteModels.jl model"
    kite::KiteModels.KPS4_3L
    "complex x0 to simple x_plus nonlinear function"
    f!::Function
    "observer function"
    h!::Function
    "linearized model of the kite"
    model::ModelPredictiveControl.NonLinModel
    mpc::ModelPredictiveControl.NonLinMPC
    optim::JuMP.Model
    U_data::Matrix{Float64}
    Y_data::Matrix{Float64}
    Ry_data::Matrix{Float64}
    X̂_data::Matrix{Float64}
    output_idxs::Vector{Int}
    observed_idxs::Vector{Int}
    s_idxs::Dict{Num, Int}
    y_noise::Vector{Float64}
    error::Float64
    plotting::Bool

    function ControlInterface(
        kite::KPS4_3L;
        Ts::Float64=0.05,
        x0::Vector{Float64}=kite.integrator.u,
        u0::Vector{Float64}=zeros(3),
        ry::Union{Nothing,Vector{Float64}}=nothing,
        noise::Float64=1e-3,
        buffer_time::Int=100,
        time_multiplier::Int=10,
    )
        x0 = copy(x0)
        # --- get symbolic input and output ---
        sys = kite.prob.f.sys
        input = [sys.set_values[i] for i in 1:3]
        output = [
            sys.heading_y
            sys.steering_angle
            sys.power_angle
            collect(sys.tether_length)
        ]
    
        s_idxs = Dict{Num, Int}()
        for (s_idx, sym) in enumerate(output)
            s_idxs[sym] = s_idx
        end
    
        # --- generate f and h functions for linearization ---
        (f!, h!, nx, nu, ny) = generate_f_h(kite, input, output, Ts)
    
        Hp, Hc = 20, 5
    
        # --- initialize output and plotting indexes ---
        if isnothing(ry)
            y0 = zeros(ny)
            h!(y0, x0, nothing, nothing)
            ry = y0
            ry[s_idxs[sys.heading_y]] = deg2rad(0.0)
            ry[s_idxs[sys.power_angle]] += 0.3
        end
        output_idxs = vcat(
            s_idxs[sys.heading_y],
            s_idxs[sys.steering_angle],
            s_idxs[sys.power_angle],
            s_idxs[sys.tether_length[1]],
            s_idxs[sys.tether_length[3]],
        )
        observed_idxs = vcat(
            # s_idxs[sys.pos[2, kite.num_A]],
            s_idxs[sys.heading_y],
            s_idxs[sys.steering_angle],
            s_idxs[sys.power_angle],
            s_idxs[sys.tether_length[1]],
            s_idxs[sys.tether_length[3]],
            # linmodel.nx + linmodel.ny
        )
    
        
        model = NonLinModel(f!, h!, Ts, nu, nx, ny, solver=nothing)
        setstate!(model, x0)

        Mwt = fill(0.0, model.ny)
        Mwt[s_idxs[sys.heading_y]] = 1e2
        Mwt[s_idxs[sys.power_angle]] = 0.0
        Mwt[s_idxs[sys.tether_length[3]]] = 1e-2
        Nwt = fill(0.0, model.nu)
        Lwt = fill(0.5, model.nu)
        @show model.nu
    
        σR = fill(1e-4, model.ny)
        σQ = fill(1e2, model.nx)
        σQint_u = fill(1, model.nu)
        nint_u = fill(1, model.nu)
        estim = ModelPredictiveControl.UnscentedKalmanFilter(model; nint_u, σQint_u, σQ, σR)
    
        optim = JuMP.Model(HiGHS.Optimizer)
        mpc = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt=1e5)
        @show kite.integrator[output]
        initstate!(mpc, u0, kite.integrator[output])
        setstate!(mpc, vcat(x0, u0))
    
        du = 20.0
        umin, umax = [-du, -du, -du], [du, du, du]
        # max = 0.5
        # Δumin, Δumax = [-max, -max, -max*10], [max, max, max*10]
        ymin = fill(-Inf, model.ny)
        ymax = fill(Inf, model.ny)
        ymin[s_idxs[sys.tether_length[1]]] = y0[s_idxs[sys.tether_length[1]]] - 4.8
        ymin[s_idxs[sys.tether_length[2]]] = y0[s_idxs[sys.tether_length[2]]] - 4.8
        ymax[s_idxs[sys.tether_length[1]]] = y0[s_idxs[sys.tether_length[1]]] - 4.5 # important: not too big!
        ymax[s_idxs[sys.tether_length[2]]] = y0[s_idxs[sys.tether_length[2]]] - 4.5
        # ymax[s_idxs[sys.tether_length[3]]] = y0[s_idxs[sys.tether_length[3]]] + 0.1
        setconstraint!(mpc; umin, umax, ymin, ymax)
    
        # --- init data ---
        N = Int(round(buffer_time / Ts)) # buffer time is the amount of time TO save
        U_data = fill(NaN, model.nu, N)
        Y_data = fill(NaN, model.ny, N)
        Ry_data = fill(NaN, model.ny, N)
        X̂_data = fill(NaN, estim.nx̂, N)
        y_noise = fill(noise, model.ny)
        error = 0.0
        plotting = true
        
        return new(
            input, output, u0, x0, ry, Ts, Hp,
            kite, f!, h!, model, 
            mpc, optim, U_data, Y_data, Ry_data, X̂_data, 
            output_idxs, observed_idxs, s_idxs,
            y_noise, error, plotting
        )
    end
end

include("mtk_interface.jl")
include("smartdiff.jl")

function lin_ulin_sim(ci::ControlInterface)
    println("linear sanity check")
    u = [-0, -50, -70]
    res = sim!(ci.model, 10, u; x0=ci.x0)
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
    x̂ = preparestate!(ci.mpc, y .+ ci.y_noise .* randn(ci.model.ny))
    u = moveinput!(ci.mpc, ry)
    # display(linearization_plot(ci, x, u))
    # setmodel!(ci.mpc, ci.model)
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
        display(controlplot(ci))
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
    model = deepcopy(ci.model)
    u = u0 .+ [0.0, 1.0, 0.0]
    x_simple_0 = copy(model.xop)
    ci.simple_h!(x_simple_0, x0)

    x_simple_plus = similar(x_simple_0)
    ci.simple_f!(x_simple_plus, x0, u, ci.Ts*n)
    lin_plus = model.A * x_simple_0 + model.Bu * u
    for _ in 1:n-1
        lin_plus = model.A * lin_plus + model.Bu * u
    end
    p = plot()
    idx = 1
    plot!(p, [x_simple_0[idx], x_simple_plus[idx]], label="nonlin")
    plot!(p, [x_simple_0[idx], lin_plus[idx]], label="lin", ylim=(-1.0, 1.0))
    return p
end

function linearize_process(ci::ControlInterface)
    linearize!(ci.model)
    setmodel!(mpc, model)
    xnext = zeros(length(x))
    ModelPredictiveControl.f!(xnext, model, x, u, nothing, nothing)
    ci.error = (norm(xnext .- x) - norm(model.fop .- x)) / norm(model.fop .- x)
    # @show norm(model.fop .- model.xop)
    # @show norm((model.A * model.xop + model.Bu * model.uop) .- model.xop)
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