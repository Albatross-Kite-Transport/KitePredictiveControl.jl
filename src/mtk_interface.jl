# function get_control_function(model, input)
#     f_ip, dvs, psym, io_sys = ModelingToolkit.generate_control_function(IRSystem(model), input)
#     # any(ModelingToolkit.is_alg_equation, equations(io_sys)) && error("Systems with algebraic equations are not supported")
#     return (f_ip, dvs, psym, io_sys)
# end

function generate_f_h(kite::KPS4_3L, input, output, Ts)
    # get_y = getu(kite.integrator, output)

    sys = kite.prob.f.sys
    state = unknowns(sys)
    nu = length(input)
    nx = length(state)
    ny = length(output)

    setu! = setp(kite.prob, [input[i] for i in 1:nu])
    # integrator = kite.integrator

    function make_default_creator(state)
        keys = collect(state)
        return x -> Dict(k => x[i] for (i, k) in enumerate(keys))
    end
    create_default = make_default_creator(state)
    solver = QBDF()
    
    integ_cache = GeneralLazyBufferCache(
        function (xu)
            x = xu[1:nx]
            u = xu[nx+1:end]
            default = create_default(x)
            par = vcat([input[i] => u[i] for i in 1:nu])
            prob = ODEProblem(sys, default, (0.0, Ts), par)
            setu! = setp(prob, [input[i] for i in 1:nu])
            get_y = getu(prob, output)
            integrator = OrdinaryDiffEq.init(prob, solver; saveat=Ts, abstol=kite.set.abs_tol, reltol=kite.set.rel_tol, verbose=false)
            return (integrator, setu!, get_y)
        end
    )

    """
    Nonlinear discrete dynamics. Takes in complex state and returns simple state_plus
    """
    function f!(x_plus, x, u, integ_cache)
        (integrator, setu!, _) = integ_cache
        reinit!(integrator, x; t0=1.0, tf=1.0 + Ts)
        setu!(integrator, u)
        @time OrdinaryDiffEq.step!(integrator, Ts, true)
        !successful_retcode(integrator.sol) && @show x_plus x u
        @assert successful_retcode(integrator.sol)
        x_plus .= integrator.u
        @show x_plus
        return nothing
    end
    function f!(x_simple_plus, x, u, _, _)
        f!(x_simple_plus, x, u, integ_cache[vcat(x, u)])
    end 

    "Observer function"
    function h!(y, x, integ_cache)
        (integrator, _, get_y) = integ_cache
        reinit!(integrator, x; t0=1.0, tf=1.0 + Ts)
        y .= get_y(integrator)
        return nothing
    end
    h!(y, x, _, _) = h!(y, x, integ_cache[vcat(x, zeros(3))])

    return (f!, h!, nx, nu, ny)
end
