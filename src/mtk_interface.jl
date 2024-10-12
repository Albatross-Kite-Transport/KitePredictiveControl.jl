
# function get_control_function(model, inputs)
#     f_ip, dvs, psym, io_sys = ModelingToolkit.generate_control_function(IRSystem(model), inputs)
#     # any(ModelingToolkit.is_alg_equation, equations(io_sys)) && error("Systems with algebraic equations are not supported")
#     return (f_ip, dvs, psym, io_sys)
# end

function generate_f_h(kite::KPS4_3L, inputs, outputs, Ts)
    nu = length(inputs)
    ny = length(outputs)
    nx = length(unknowns(kite.simple_sys))
    vu = string.(inputs)
    vy = string.(outputs)
    vx = string.(unknowns(kite.simple_sys))
    # par = initial_conditions(io_sys, defaults(io_sys), psym)[2]
    
    # @show h_(zeros(ny), zeros(nx), kite.integrator.p, 1.0)

    get_out = getu(kite.integrator.sol, outputs)
    

    # function loss(p)
    #     _prob = remake(prob, u0 = eltype(p).(prob.u0), p = MyStruct(p))
    #     sol = solve(_prob, ...)
    #     # do stuff on sol
    # end
    solver = QNDF()
    function f(state, input, _, _)
        # OrdinaryDiffEq.reinit!(kite.integrator, state)
        # kite.set_set_values(kite.prob, input)
        prob = remake(kite.prob; u0 = state, p = [kite.simple_sys.set_values => input])
        sol = OrdinaryDiffEq.solve(prob, solver; saveat=0.1, abstol=1e-2, reltol=1e-2);
        return @views sol.u[end]
    end
    function h(y, x, _, _)
        return get_out(kite.integrator)
    end
    return f, (h, nu, ny, nx, vu, vy, vx)
end