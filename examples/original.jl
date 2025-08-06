using ModelPredictiveControl, ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t, varmap_to_vars
@mtkmodel Pendulum begin
    @parameters begin
        g = 9.8
        L = 0.4
        K = 1.2
        m = 0.3
    end
    @variables begin
        θ(t) # state
        ω(t) # state
        τ(t) # input
        y(t) # output
    end
    @equations begin
        D(θ)    ~ ω
        D(ω)    ~ -g/L*sin(θ) - K/m*ω + τ/m/L^2
        y       ~ θ * 180 / π
    end
end
@named mtk_model = Pendulum()
mtk_model = complete(mtk_model)

function generate_f_h(model, inputs, outputs)
    (_, f_ip), dvs, psym, io_sys = ModelingToolkit.generate_control_function(
        model, inputs, split=false; outputs
    )
    (h_ip, h_oop) = ModelingToolkit.build_explicit_observed_function(
        io_sys, outputs; inputs, return_inplace = true
    )
    return f_ip, h_ip, h_oop
end
inputs, outputs = [mtk_model.τ], [mtk_model.y]
f_ip, h_ip, h_oop = generate_f_h(mtk_model, inputs, outputs)
ny, nx, nu = length(outputs), length(unknowns(mtk_model)), length(inputs)
