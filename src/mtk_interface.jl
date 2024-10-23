MTK = ModelingToolkit
# function get_control_function(model, inputs)
#     f_ip, dvs, psym, io_sys = ModelingToolkit.generate_control_function(IRSystem(model), inputs)
#     # any(ModelingToolkit.is_alg_equation, equations(io_sys)) && error("Systems with algebraic equations are not supported")
#     return (f_ip, dvs, psym, io_sys)
# end

function generate_f_h(kite::KPS4_3L, inputs, outputs, Ts)
    get_y = getu(kite.integrator, outputs)
    nu = length(inputs)
    nx = length(kite.integrator.u)
    ny = length(outputs)

    sys = kite.prob.f.sys
    solver = OrdinaryDiffEqBDF.QBDF()

    function make_default_creator(sys)
        keys = collect(unknowns(sys))
        return x -> Dict(k => x[i] for (i, k) in enumerate(keys))
    end
    create_default = make_default_creator(sys)

    integ_cache = GeneralLazyBufferCache(
        function (xu)
            x = xu[1:nx]
            u = xu[nx+1:end]
            default = create_default(x)
            par = vcat([inputs[i] => u[i] for i in 1:nu])
            prob = ODEProblem(sys, default, (0.0, Ts), par)
            setu! = setp(prob, [inputs[i] for i in 1:nu])
            integrator = OrdinaryDiffEqCore.init(prob, solver; saveat=Ts, abstol=kite.set.abs_tol, reltol=kite.set.rel_tol, verbose=false)
            return (integrator, setu!)
        end
    )

    "Nonlinear discrete dynamics"
    function next_step!(x_plus, x, u, integ_setu_pair)
        (integ, setu!) = integ_setu_pair
        reinit!(integ, x; t0=1.0, tf=1.0+Ts)
        setu!(integ, u)
        OrdinaryDiffEqCore.step!(integ, Ts, true)
        @assert successful_retcode(integ.sol)
        x_plus .= integ.u
        return nothing
    end
    f!(x_plus, x, u, _, _) = next_step!(x_plus, x, u, integ_cache[vcat(x, u)])

    "Observer function"
    function get_y!(y, x, integ_setu_pair)
        (integ, _) = integ_setu_pair
        reinit!(integ, x; t0=1.0, tf=1.0+Ts)
        y .= get_y(integ)
        return nothing
    end
    h!(y, x, _, _) = get_y!(y, x, integ_cache[vcat(x, zeros(3))])

    return f!, h!, nu, nx, ny
end

function get_next_state(kite::KPS4_3L, x, u, Ts)
    OrdinaryDiffEqCore.reinit!(kite.integrator, x; t0=1.0, tf=1.0+Ts)
    next_step!(kite; set_values = u, dt = Ts)
    return kite.integrator.u
end

function linearize!(ci::ControlInterface, linmodel, x, u, p)
    return linearize(ci.kite, ci.sys, ci.lin_fun, ci.get_y, x, u, p, ci.Ts; linmodel)
end
function linearize(kite::KPS4_3L, sys, lin_fun, get_y, x::Vector{Float64}, 
            u::Vector{Float64}, p::MTKParameters, Ts; linmodel = nothing, t = 1.0, allow_input_derivatives = false)
    linres = lin_fun(x, p, t)
    f_x, f_z, g_x, g_z, f_u, g_u, h_x, h_z, h_u = linres
    nx, nu = size(f_u)
    ny = size(h_x, 1)
    D = h_u
    if isempty(g_z)
        A = f_x
        B = f_u
        C = h_x
        @assert iszero(g_x)
        @assert iszero(g_z)
        @assert iszero(g_u)
    else
        gz = lu(g_z; check = false)
        issuccess(gz) ||
            error("g_z not invertible, this indicates that the DAE is of index > 1.")
        gzgx = -(gz \ g_x)
        A = [f_x f_z
            gzgx*f_x gzgx*f_z]
        B = [f_u
            gzgx * f_u] # The cited paper has zeros in the bottom block, see derivation in https://github.com/SciML/ModelingToolkit.jl/pull/1691 for the correct formula
        C = [h_x h_z]
        Bs = -(gz \ g_u) # This equation differ from the cited paper, the paper is likely wrong since their equaiton leads to a dimension mismatch.
        if !iszero(Bs)
            if !allow_input_derivatives
                der_inds = findall(vec(any(!=(0), Bs, dims = 1)))
                error("Input derivatives appeared in expressions (-g_z\\g_u != 0), the following inputs appeared differentiated: $(inputs(sys)[der_inds]). Call `linearize` with keyword argument `allow_input_derivatives = true` to allow this and have the returned `B` matrix be of double width ($(2nu)), where the last $nu inputs are the derivatives of the first $nu inputs.")
            end
            B = [B [zeros(nx, nu); Bs]]
            D = [D zeros(ny, nu)]
        end
    end
    # --- discretize and generate model ---
    css = ss(A, B, C, D)
    dss = c2d(css, Ts, :zoh)
    if isnothing(linmodel)
        linmodel = LinModel(dss.A, dss.B, dss.C, dss.B[:, end+1:end], dss.D[:, end+1:end], Ts)
    else
        linmodel.A  .= dss.A
        linmodel.Bu .= dss.B
        linmodel.C  .= dss.C
    end
    # --- modify the linear model operating points ---
    linmodel.uop .= u
    linmodel.yop .= get_y(kite.integrator)
    linmodel.xop .= x
    linmodel.fop .= get_next_state(kite, x, u, Ts)

    der_x = css.A * x + css.B * u
    x_next_lin = dss.A * x + dss.B * u
    @show Ts
    @show norm(x .- linmodel.fop)
    @show norm(x .- x_next_lin)
    @show norm(der_x)
    @show u
    # @show x
    @assert false
    # --- reset the state of the linear model ---
    linmodel.x0 .= 0 # state deviation vector is always x0=0 after a linearization
    return linmodel
end

function ModelingToolkit.variable_index(name::Vector{String}, var)
    return findfirst(x -> x == string(var), name)
end

# 532219