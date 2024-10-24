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

function ModelingToolkit.variable_index(name::Vector{String}, var)
    return findfirst(x -> x == string(var), name)
end
