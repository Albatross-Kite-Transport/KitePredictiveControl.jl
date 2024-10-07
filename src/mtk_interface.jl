
function get_control_function(model, inputs)
    f_ip, dvs, psym, io_sys = ModelingToolkit.generate_control_function(IRSystem(model), inputs)
    # any(ModelingToolkit.is_alg_equation, equations(io_sys)) && error("Systems with algebraic equations are not supported")
    return (f_ip, dvs, psym, io_sys)
end

function generate_f_h(sys, inputs, outputs, f_ip, dvs, psym)
    h_ = JuliaSimCompiler.build_explicit_observed_function(sys, outputs; inputs = inputs, target = JuliaSimCompiler.JuliaTarget())
    nu = length(inputs)
    ny = length(outputs)
    nx = length(dvs)
    vu = string.(inputs)
    vy = string.(outputs)
    vx = string.(dvs)
    par = JuliaSimCompiler.initial_conditions(io_sys, defaults(io_sys), psym)[2]
    # include("../f.jl")
    function f!(dx, x, u, _, _)
        # @assert length(dx) == length(x)
        f_ip(dx, x, u, par, 1.0)
        if !all(isfinite.(dx))
            dx .= 0.0
            # Core.println("x ", x)
            # Core.println("dx ", dx)
            # Core.println("u ", u)
            # Core.println("par ", par)
        end
        nothing
    end
    function h!(y, x, _, _)
        h_(y, x, fill(nothing, length(inputs)), par, 1.0)
        nothing
    end
    return f!, (h!, nu, ny, nx, vu, vy, vx)
end