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
export step!, controlplot, start_processes!, stop_processes!

set_data_path(joinpath(@__DIR__, "..", "data"))
mutable struct ControlInterface
    "symbolic inputs"
    inputs::Vector{Num}
    "symbolic outputs"
    outputs::Vector{Num}
    "initial inputs"
    u0::Vector{Float64}
    "initial state"
    x0::Vector{Float64}
    "wanted outputs"
    ry::Vector{Float64}
    "sampling time [seconds]"
    Ts::Float64
    "predict horizon [number of steps]"
    Hp::Int
    "nonlinear KiteModels.jl model"
    kite::KiteModels.KPS4_3L
    "complex x0 to simple x_plus nonlinear function"
    measure_f!::Function
    simple_f!::Function
    "observer function"
    simple_h!::Function
    "linearized model of the kite"
    linmodel::ModelPredictiveControl.LinModel
    "measurement matrixes X_plus, X and U"
    measurements::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
    mpc::ModelPredictiveControl.LinMPC
    optim::JuMP.Model
    U_data::Matrix{Float64}
    Y_data::Matrix{Float64}
    Ry_data::Matrix{Float64}
    X̂_data::Matrix{Float64}
    output_idxs::Vector{Int}
    observed_idxs::Vector{Int}
    s_idxs::Dict{Num, Int}
    m_idxs::Dict{Num, Int}
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
        buffer_time::Int=20,
        time_multiplier::Int=10,
    )
        x0 = copy(x0)
        # --- get symbolic inputs and outputs ---
        sys = kite.prob.f.sys
        inputs = [sys.set_values[i] for i in 1:3]
        measure_state = vcat(
            sys.heading_y,
            sys.turn_rate_y,
            sys.flap_diff,
            sys.flap_diff_vel,
            sys.tether_diff,
            sys.tether_diff_vel,
            vcat(sys.tether_length),
            vcat(sys.tether_vel)
        )
        simple_state = vcat(
            sys.heading_y,
            sys.flap_diff,
            vcat(sys.tether_length),
        )
        outputs = simple_state
    
        s_idxs = Dict{Num, Int}()
        for (s_idx, sym) in enumerate(simple_state)
            s_idxs[sym] = s_idx
        end
        m_idxs = Dict{Num, Int}()
        for (m_idx, sym) in enumerate(measure_state)
            m_idxs[sym] = m_idx
        end
    
        # --- generate f and h functions for linearization ---
        (simple_f!, measure_f!, simple_h!, simple_state, nu, nsx, nmx) = generate_f_h(kite, simple_state, measure_state, inputs, Ts)
    
        # --- linearize model ---
        Hp, Hc = 40, 1
        linmodel, measurements = linearize(sys, s_idxs, m_idxs, measure_f!, simple_f!, simple_h!, nsx, nmx, nu,
            string.(simple_state), string.(inputs),
            x0, u0, Ts, Hp)
    
        # --- initialize outputs and plotting indexes ---
        if isnothing(ry)
            y0 = zeros(nsx)
            simple_h!(y0, x0)
            ry = y0
            ry[s_idxs[sys.heading_y]] = deg2rad(0.0)
        end
        output_idxs = vcat(
            s_idxs[sys.heading_y],
            s_idxs[sys.flap_diff],
            s_idxs[sys.tether_length[1]],
            s_idxs[sys.tether_length[2]],
            s_idxs[sys.tether_length[3]],
        )
        observed_idxs = vcat(
            # s_idxs[sys.pos[2, kite.num_A]],
            s_idxs[sys.heading_y],
            s_idxs[sys.flap_diff],
            s_idxs[sys.tether_length[1]],
            s_idxs[sys.tether_length[2]],
            s_idxs[sys.tether_length[3]],
            # linmodel.nx + linmodel.ny
        )
    
        Mwt = fill(0.0, linmodel.ny)
        Mwt[s_idxs[sys.heading_y]] = 1.0
        # Mwt[s_idxs[sys.tether_length[1])] = 0.1
        # Mwt[s_idxs[sys.tether_length[2])] = 0.1
        Mwt[s_idxs[sys.tether_length[3]]] = 1.0
        Nwt = fill(0.0, linmodel.nu)
        Lwt = fill(0.1, linmodel.nu)
    
        σR = fill(1e-4, linmodel.ny)
        σQ = fill(1e2, linmodel.nx)
        σQint_u = fill(1, linmodel.nu)
        nint_u = fill(1, linmodel.nu)
        estim = ModelPredictiveControl.UnscentedKalmanFilter(linmodel; nint_u, σQint_u, σQ, σR)
    
        optim = JuMP.Model(HiGHS.Optimizer)
        mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt=Inf)
    
        umin, umax = [-10, -10, -10], [10, 10, 10] # TODO: torque control with u0 = -winch_forces * drum_radius
        # max = 0.5
        # Δumin, Δumax = [-max, -max, -max*10], [max, max, max*10]
        ymin = fill(-Inf, linmodel.ny)
        ymax = fill(Inf, linmodel.ny)
        ymin[s_idxs[sys.tether_length[1]]] = y0[s_idxs[sys.tether_length[1]]] - 1.0
        ymin[s_idxs[sys.tether_length[2]]] = y0[s_idxs[sys.tether_length[2]]] - 1.0
        setconstraint!(mpc; umin, umax, ymin, ymax)
        # initstate!(mpc, zeros(3), y0) # TODO: check if needed
    
        # --- init data ---
        N = Int(round(buffer_time / Ts)) # buffer time is the amount of time to save
        U_data = fill(NaN, linmodel.nu, N)
        Y_data = fill(NaN, linmodel.ny, N)
        Ry_data = fill(NaN, linmodel.ny, N)
        X̂_data = fill(NaN, estim.nx̂, N)
        y_noise = fill(noise, linmodel.ny)
        error = 0.0
        plotting = true
        
        return new(
            inputs, outputs, u0, x0, ry, Ts, Hp,
            kite, measure_f!, simple_f!, simple_h!, linmodel, measurements, 
            mpc, optim, U_data, Y_data, Ry_data, X̂_data, 
            output_idxs, observed_idxs, s_idxs, m_idxs,
            y_noise, error, plotting
        )
    end
end




include("mtk_interface.jl")
include("smartdiff.jl")

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
    # display(linearization_plot(ci, x, u))
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
function linearization_plot(ci::ControlInterface, x0, u0; n::Int=10)
    linmodel = deepcopy(ci.linmodel)
    u = u0 .+ [0.0, 1.0, 0.0]
    x_simple_0 = copy(linmodel.xop)
    ci.simple_h!(x_simple_0, x0)

    x_simple_plus = similar(x_simple_0)
    ci.simple_f!(x_simple_plus, x0, u, ci.Ts*n)
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