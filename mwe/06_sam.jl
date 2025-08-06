using SymbolicAWEModels
using ModelingToolkit
using ModelPredictiveControl

include("generate_f_h.jl")

REMAKE = 1

if !@isdefined(plant) || Bool(REMAKE)
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
end

using Plots
u = zeros(3)
N = 10
res = ModelPredictiveControl.sim!(model, N, u)
plot(res, plotu=false)
