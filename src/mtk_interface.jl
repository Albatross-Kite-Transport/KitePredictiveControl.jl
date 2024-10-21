
function get_control_function(model, inputs)
    f_ip, dvs, psym, io_sys = ModelingToolkit.generate_control_function(IRSystem(model), inputs)
    # any(ModelingToolkit.is_alg_equation, equations(io_sys)) && error("Systems with algebraic equations are not supported")
    return (f_ip, dvs, psym, io_sys)
end

function generate_f_h(sys, inputs, outputs, f_ip, dvs, psym, x_0)
    h_ = JuliaSimCompiler.build_explicit_observed_function(sys, outputs; inputs = inputs, target = JuliaSimCompiler.JuliaTarget())
    nu = length(inputs)
    ny = length(outputs)
    nx = length(dvs)
    vu = string.(inputs)
    vy = string.(outputs)
    vx = string.(dvs)
    par = JuliaSimCompiler.initial_conditions(io_sys, defaults(io_sys), psym)[2]
    function f_oop(x, u, par, t)
        dx = Vector{Any}(undef, nx)
        f_ip(dx, x, u, par, t)
        return dx
    end
    # fails when using SimpleColloc, but works when using Rk4
    # f_disc = SeeToDee.Rk4(f_oop, Ts; supersample = 1)
    f_disc = SimpleColloc(f_oop, Ts, nx, 0, nu; n=40, abstol=1e-4) 
    println("linearize")
    @time SeeToDee.linearize(f_disc, x_0, zeros(3), par, 1.0)
    @time SeeToDee.linearize(f_disc, x_0, zeros(3), par, 1.0)
    function f(x, u, _, _)
        return f_disc(x, u, par, 1.0)
    end
    function h!(y, x, _, _)
        h_(y, x, fill(nothing, length(inputs)), par, 1.0)
        nothing
    end
    return f, (h!, nu, ny, nx, vu, vy, vx)
end