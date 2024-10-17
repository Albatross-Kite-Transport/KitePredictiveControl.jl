MTK = ModelingToolkit
# function get_control_function(model, inputs)
#     f_ip, dvs, psym, io_sys = ModelingToolkit.generate_control_function(IRSystem(model), inputs)
#     # any(ModelingToolkit.is_alg_equation, equations(io_sys)) && error("Systems with algebraic equations are not supported")
#     return (f_ip, dvs, psym, io_sys)
# end

function generate_f_h(kite::KPS4_3L, inputs, outputs, Ts)
    get_out = getu(kite.integrator.sol, outputs)
    solver = QNDF(autodiff=false)

    function f!(next_state, state, input, _, _)
        # @show kite.prob.u0
        # prob = remake(kite.prob; u0 = state, p = [kite.simple_sys.set_values => input], tspan=(0.0, Ts))
        # @show prob.u0
        # sol = OrdinaryDiffEqCore.solve(prob, solver; saveat=0.1, abstol=kite.set.abs_tol, reltol=kite.set.rel_tol);
        OrdinaryDiffEqCore.reinit!(kite.integrator, state)
        kite.set_set_values(kite.integrator, input)
        OrdinaryDiffEqCore.step!(kite.integrator, Ts, true)
        next_state .= kite.integrator.u
        nothing
    end
    function h!(outputs, state, _, _)
        OrdinaryDiffEqCore.reinit!(kite.integrator, state)
        outputs .= get_out(kite.integrator)
        nothing
    end
    return f!, h!
end

function linearize(sys, lin_fun, u0::Vector{Float64}, p::MTKParameters; t = 1.0, allow_input_derivatives = false)
    linres = lin_fun(u0, p, t)
    f_x, f_z, g_x, g_z, f_u, g_u, h_x, h_z, h_u = linres

    nx, nu = size(f_u)
    nz = size(f_z, 2)
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

    (; A, B, C, D)
end

function ModelingToolkit.variable_index(name::Vector{String}, var)
    return findfirst(x -> x == string(var), name)
end