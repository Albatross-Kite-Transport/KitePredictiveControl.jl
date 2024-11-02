# function get_control_function(model, inputs)
#     f_ip, dvs, psym, io_sys = ModelingToolkit.generate_control_function(IRSystem(model), inputs)
#     # any(ModelingToolkit.is_alg_equation, equations(io_sys)) && error("Systems with algebraic equations are not supported")
#     return (f_ip, dvs, psym, io_sys)
# end

function generate_f_h(kite::KPS4_3L, simple_state, measure_state, inputs, Ts)
    get_y = getu(kite.integrator, simple_state)

    sys = kite.prob.f.sys
    state = unknowns(sys)
    nu = length(inputs)
    nx_simple = length(simple_state)
    nx_measure = length(measure_state)
    nx = length(state)

    setu! = setp(kite.prob, [inputs[i] for i in 1:nu])
    get_simple_x = getu(kite.prob, simple_state)
    get_measure_x = getu(kite.prob, measure_state)
    integrator = kite.integrator

    """
    Nonlinear discrete dynamics. Takes in complex state and returns simple state_plus
    """
    function step!(x, u, dt, integ_setu_pair)
        (integ, setu!) = integ_setu_pair
        reinit!(integ, x; t0=1.0, tf=1.0 + dt)
        uop = -kite.get_winch_forces(integ) * kite.set.drum_radius
        setu!(integ, u .+ uop)
        OrdinaryDiffEq.step!(integ, dt, true)
        @assert successful_retcode(integ.sol)
        return nothing
    end
    function simple_f!(x_simple_plus, x, u, dt)
        step!(x, u, dt, (integrator, setu!))
        x_simple_plus .= get_simple_x(integrator)
        return x_simple_plus
    end
    function measure_f!(x_measure_plus, x, u, dt)
        step!(x, u, dt, (integrator, setu!))
        x_measure_plus .= get_measure_x(integrator)
        return x_measure_plus
    end

    "Observer function"
    function get_y!(y, x, integ_setu_pair)
        (integ, _, _) = integ_setu_pair
        reinit!(integ, x; t0=1.0, tf=1.0 + Ts)
        y .= get_y(integ)
        return y
    end
    simple_h!(y, x) = get_y!(y, x, (integrator, setu!, get_simple_x))

    return (simple_f!, measure_f!, simple_h!, simple_state, nu, nx_simple, nx_measure)
end
