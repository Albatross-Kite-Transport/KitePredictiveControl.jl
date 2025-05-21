struct ModelParams{G1,G2}
    s::RamAirKite
    integ::OrdinaryDiffEq.ODEIntegrator
    set_x::Union{SymbolicIndexingInterface.MultipleSetters, Nothing}
    set_ix::Union{SymbolicIndexingInterface.MultipleSetters, Nothing}
    set_u::SymbolicIndexingInterface.MultipleSetters
    get_x::G1
    get_y::G2
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
    function ModelParams(s, y_vec)
        # [println(x) for x in x_vec]
        # y_vec = [
        #     [s.sys.tether_length[i] for i in 1:3]
        #     s.sys.elevation
        #     s.sys.azimuth
        # ]
        x_vec = KiteModels.get_unknowns(s_model)
        u_vec = [s.sys.set_values[i] for i in 1:3]

        set_x = setu(s.integrator, x_vec)
        set_ix = setu(s.integrator, Initial.(x_vec))
        set_u = setu(s.integrator, u_vec)
        get_x = getu(s.integrator, x_vec)
        get_y = getu(s.integrator, y_vec)
        x0 = get_x(s.integrator)
        y0 = get_y(s.integrator)
        u0 = -s.set.drum_radius * s.integrator[s.sys.winch_force]
        u0 = [-50.0, -1.0, -1.0]
        x_idxs = Dict{Num, Int}()
        for (idx, sym) in enumerate(x_vec)
            x_idxs[sym] = idx
        end
        y_idxs = Dict{Num, Int}()
        for (idx, sym) in enumerate(y_vec)
            y_idxs[sym] = idx
        end
        Q_idxs = [x_idxs[s.sys.Q_b_w[i]] for i in 1:4]
        
        return new{typeof(get_x), typeof(get_y)}(
            s, s.integrator, set_x, set_ix, set_u, 
            get_x, get_y, dt, x_vec, y_vec, u_vec,
            x0, y0, u0, x_idxs, y_idxs, Q_idxs
        )
    end
end

