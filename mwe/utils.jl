
struct ModelParams{G1,G2,G3}
    s::RamAirKite
    integ::OrdinaryDiffEq.ODEIntegrator
    set_x::Union{SymbolicIndexingInterface.MultipleSetters, Nothing}
    set_ix::Union{SymbolicIndexingInterface.MultipleSetters, Nothing}
    set_u::SymbolicIndexingInterface.MultipleSetters
    set_state::SymbolicIndexingInterface.MultipleSetters
    get_x::G1
    get_y::G2
    get_state::G3
    dt::Float64
    
    x_vec::Vector{Num}
    y_vec::Vector{Num}
    u_vec::Vector{Num}
    x0::Vector{Float64}
    y0::Vector{Float64}
    u0::Vector{Float64}
    x_idxs::Dict{Num, Int64}
    y_idxs::Dict{Num, Int64}
    Q_idxs::Vector{Int64}
    function ModelParams(s, y_vec, x0=nothing, y0=nothing, u0=nothing)
        # [println(x) for x in x_vec]
        # y_vec = [
        #     [s.sys.tether_length[i] for i in 1:3]
        #     s.sys.elevation
        #     s.sys.azimuth
        # ]
        x_vec = ModelingToolkit.unknowns(s.sys)
        u_vec = [s.sys.set_values[i] for i in 1:3]

        set_x = setu(s.integrator, x_vec)
        set_ix = setu(s.integrator, Initial.(x_vec))
        set_u = setu(s.integrator, u_vec)
        set_state = setu(s.integrator, 
            [sys.kite_pos, sys.kite_vel, sys.Q_b_w, sys.ω_b, sys.tether_length, sys.tether_vel]
        )
        
        get_x = getu(s.integrator, x_vec)
        get_y = getu(s.integrator, y_vec)
        get_state = getu(s.integrator, s.sys.distance)
        isnothing(x0) && (x0 = get_x(s.integrator))
        isnothing(y0) && (y0 = get_y(s.integrator))
        isnothing(u0) && (u0 = [-50.0, -1.0, -1.0])
        x_idxs = Dict{Num, Int}()
        for (idx, sym) in enumerate(x_vec)
            x_idxs[sym] = idx
        end
        y_idxs = Dict{Num, Int}()
        for (idx, sym) in enumerate(y_vec)
            y_idxs[sym] = idx
        end
        Q_idxs = [x_idxs[s.sys.Q_b_w[i]] for i in 1:4]
        
        return new{typeof(get_x),typeof(get_y),typeof(get_state)}(
            s, s.integrator, set_x, set_ix, set_u, set_state, 
            get_x, get_y, get_state, dt, x_vec, y_vec, u_vec,
            x0, y0, u0, x_idxs, y_idxs, Q_idxs
        )
    end
end

