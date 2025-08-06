using SymbolicAWEModels
using ModelingToolkit
using Suppressor

if !ispath("data")
    SymbolicAWEModels.copy_model_settings()
end

function generate_f_h(model, inputs, outputs)
    (f_oop, f_ip), dvs, psym, io_sys = @suppress_err ModelingToolkit.generate_control_function(
        model, inputs; simplify=false
    )
    nu, nx, ny = length(inputs), length(dvs), length(outputs)
    (h_oop, h_ip) = ModelingToolkit.build_explicit_observed_function(
        io_sys, outputs; inputs, return_inplace = true, eval_expression=true
    )
    return f_oop, f_ip, h_oop, h_ip, nu, nx, ny, psym
end

set = Settings("system.yaml")
simple_sam = SymbolicAWEModel(set, "simple_ram")
init!(simple_sam)
full_sys = complete(simple_sam.full_sys)

inputs, outputs = [full_sys.set_values...], [simple_sam.full_sys.heading[1]]
@time f_oop, f_ip, h_oop, h_ip, nu, nx, ny, psym = generate_f_h(full_sys, inputs, outputs)
p = simple_sam.integrator.ps[psym]
@show typeof.(p)
f_ip(zeros(nx), zeros(nx), zeros(nu), p, 0.0)
f_oop(zeros(nx), zeros(nu), p, 0.0)
