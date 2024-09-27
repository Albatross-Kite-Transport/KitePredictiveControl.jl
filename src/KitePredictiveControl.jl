module KitePredictiveControl

using ModelPredictiveControl
using ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t
using KiteModels
using Plots
using Serialization

export run_controller

# function __init__()
#     # 20 minutes
#     @eval function Serialization.resolve_ref_immediately(s::Serialization.AbstractSerializer, x)
#         s.table[s.counter] = x
#         s.counter += 1
#         nothing
#     end
# end

include("mtk_interface.jl")


function run_controller()
    set_data_path(joinpath(pwd(), "data"))
    kite::KPS4_3L = KPS4_3L(KCU(se("system_3l.yaml")))
    kite.torque_control = true
    pos, vel = init_pos_vel(kite)
    kite_model, inputs = KiteModels.model!(kite, pos, vel)
    outputs = []

    println("get_control_function") # started 8:42
    @time (f_ip, dvs, psym, kite_model) = get_control_function(kite_model, inputs; filename="kite_control_function.bin")

    

    # println("generating f and h")
    # @time f!, h!, nx, vx = generate_f_h(inputs, outputs, f_ip, dvs, psym, kite_model)
    # nu, ny, Ts = 1, 1, 0.1
    # vu, vy = ["\$τ\$ (Nm)"], ["\$θ\$ (°)"]
    # println("NonLinModel")
    # @time model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny); u=vu, x=vx, y=vy)
    # println("LinModel")
    # @time linmodel = linearize(model)
    # @time linmodel = linearize(model)
    # @time linmodel = linearize(model)

    # u = [0.5]
    # N = 35
    # res = sim!(model, N, u)
    # display(plot(res, plotu=false))

    # α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
    # estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)

    # # kite_model.K = defaults(kite_model)[kite_model.K] * 1.25
    # f_plant, h_plant, _, _ = generate_f_h(kite_model, inputs, outputs)
    # plant = setname!(NonLinModel(f_plant, h_plant, Ts, nu, nx, ny); u=vu, x=vx, y=vy)
    # println("sim")
    # @time res = sim!(estim, N, [0.5], plant=plant, y_noise=[0.5])
    # display(plot(res, plotu=false, plotxwithx̂=true))

    # Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]
    # nmpc = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
    # umin, umax = [-1.5], [+1.5]
    # nmpc = setconstraint!(nmpc; umin, umax)

    # res_ry = sim!(nmpc, N, [180.0], plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
    # display(plot(res_ry))

    # res_yd = sim!(nmpc, N, [180.0], plant=plant, x_0=[π, 0], x̂_0=[π, 0, 0], y_step=[10])
    # display(plot(res_yd))
    nothing
end

# @setup_workload begin
#     @compile_workload begin
#         run_controller()
#         nothing
#     end
# end

end