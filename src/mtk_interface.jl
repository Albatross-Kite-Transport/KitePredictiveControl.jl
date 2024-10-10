
function get_control_function(model, inputs)
    f_ip, dvs, psym, io_sys = ModelingToolkit.generate_control_function(IRSystem(model), inputs)
    # any(ModelingToolkit.is_alg_equation, equations(io_sys)) && error("Systems with algebraic equations are not supported")
    return (f_ip, dvs, psym, io_sys)
end

function generate_f_h(kite::KPS4_3L, io_sys, inputs, outputs, f_ip, dvs, psym, Ts)
    h_ = JuliaSimCompiler.build_explicit_observed_function(io_sys, outputs; inputs = inputs, target = JuliaSimCompiler.JuliaTarget())
    nu = length(inputs)
    ny = length(outputs)
    nx = length(dvs)
    vu = string.(inputs)
    vy = string.(outputs)
    vx = string.(dvs)
    par = JuliaSimCompiler.initial_conditions(io_sys, defaults(io_sys), psym)[2]
    # function f_oop(x, u, par, t)
    #     dx = Vector{Any}(undef, nx)
    #     f_ip(dx, x, u, par, t)
    #     return dx
    # end
    # f_disc = SeeToDee.Rk4(f_oop, Ts; supersample = Int(5e3))
    # f_disc = SimpleColloc(f_oop, Ts, nx, 0, nu; n = 5, abstol = 1e-8, solver=NewtonRaphson(), residual=false)
    # SeeToDee.linearize(f_disc, x_0, ones(3), par, 10.0)

    solver = QNDF()
    function f(x, u, _, _)
        @time OrdinaryDiffEq.reinit!(kite.integrator, x)
        next_step!(kite; set_values=u, dt=Ts)
        return f_disc(x, u, par, 1.0)
    end
    # @assert false
    function h!(y, x, _, _)
        h_(y, x, fill(nothing, length(inputs)), par, 1.0)
        nothing
    end
    return f, (h!, nu, ny, nx, vu, vy, vx) # TODO: check on mwe.jl
end