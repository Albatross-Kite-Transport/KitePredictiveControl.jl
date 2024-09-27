
function get_control_function(model, inputs; filename="control_function.bin")
    if isfile(filename)
        println("deserializing control function")
        return deserialize(filename)
    else
        println("generating control function: sit back, relax and enjoy...")
        (_, f_ip), dvs, psym, io_sys = ModelingToolkit.generate_control_function(model, inputs; split=false, simplify=true)
        any(ModelingToolkit.is_alg_equation, equations(io_sys)) && error("Systems with algebraic equations are not supported")
        println("serializing control function")
        serialize(filename, (f_ip, dvs, psym, io_sys))
        return (f_ip, dvs, psym, io_sys)
    end
end

function generate_f_h(inputs, outputs, f_ip, dvs, psym, io_sys)
    h_ = ModelingToolkit.build_explicit_observed_function(io_sys, outputs; inputs = inputs)
    nx = length(dvs)
    vx = string.(dvs)
    par = varmap_to_vars(defaults(io_sys), psym)
    function f!(dx, x, u, _)
        f_ip(dx, x, u, par, 1)
        nothing
    end
    function h!(y, x, _)
        y .= h_(x, 1, par, 1)
        nothing
    end
    return f!, h!, nx, vx
end