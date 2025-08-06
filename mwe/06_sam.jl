using SymbolicAWEModels
using ModelingToolkit
using ModelPredictiveControl

using Suppressor

function generate_f_h(model, inputs, outputs)
    (_, f_ip), dvs, psym, io_sys = @suppress_err ModelingToolkit.generate_control_function(
        model, inputs; simplify=false
    )
    global f_ip, h_ip, h_oop
    if any(ModelingToolkit.is_alg_equation, equations(io_sys))
        error("Systems with algebraic equations are not supported")
    end
    nu, nx, ny = length(inputs), length(dvs), length(outputs)
    vx = string.(dvs)
    function f!(ẋ, x, u, _ , p)
        try
            f_ip(ẋ, x, u, p, nothing)
        catch err
            if err isa MethodError
                error("NonLinModel does not support a time argument t in the f function, "*
                      "see the constructor docstring for a workaround.")
            else
                rethrow()
            end
        end
        return nothing
    end
    (h_oop, h_ip) = ModelingToolkit.build_explicit_observed_function(
        io_sys, outputs; inputs, return_inplace = true, eval_expression=true
    )
    u_nothing = fill(nothing, nu)
    function h!(y, x, _ , p)
        try
            # MTK.jl supports a `u` and `t` argument in `h_ip` function but not this package. We set
            # `u` as a vector of nothing and `h_ip` function will presumably throw an
            # MethodError it this argument is used inside the function
            h_ip(y, x, u_nothing, p, nothing)
        catch err
            if err isa MethodError
                error("NonLinModel only support strictly proper systems (no manipulated "*
                      "input argument u in the output function h)")
            else
                rethrow()
            end
        end
        return nothing
    end
    return f!, h!, psym, nu, nx, ny, vx
end

# REMAKE = 1
# if !@isdefined(plant) || Bool(REMAKE)
set = Settings("system.yaml")
simple_sam = SymbolicAWEModel(set, "simple_ram")
init!(simple_sam)
full_sys = complete(simple_sam.full_sys)

inputs, outputs = [full_sys.set_values...], [simple_sam.full_sys.heading[1]]
@time f!, h!, psym, nu, nx, ny, vx = generate_f_h(full_sys, inputs, outputs)
Ts = 0.1
vu, vy = ["\$τ1\$ (Nm)", "\$τ2\$ (Nm)", "\$τ3\$ (Nm)"], ["\$heading\$ (rad)"]

p = simple_sam.integrator.ps[psym]
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p); u=vu, x=vx, y=vy)
plant = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p); u=vu, x=vx, y=vy)
# end

using Plots
u = zeros(3)
N = 10
res = ModelPredictiveControl.sim!(model, N, u)
plot(res, plotu=false)
