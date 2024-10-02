
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
    function f!(dx, x, u, _, _)
        if isa(u, Vector) && u[1] != 0.0
            @show u
        end
        try
            f_ip(dx, x, u, par, 0.0)
        catch e
            # @show dx x u
            # rethrow(e)
        end
        nothing
    end
    function h!(y, x, _, _)
        h_(y, x, fill(nothing, length(inputs)), par, nothing)
        nothing
    end
    return f!, (h!, nu, ny, nx, vu, vy, vx)
end