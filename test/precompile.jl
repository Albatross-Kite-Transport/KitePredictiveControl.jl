using ModelPredictiveControl
using ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t
using KiteModels
using Serialization
using Plots

include("../src/mtk_interface.jl")

set_data_path(joinpath(pwd(), "data"))
kite::KPS4_3L = KPS4_3L(KCU(se("system_3l.yaml")))
kite.torque_control = true
pos, vel = init_pos_vel(kite)
kite_model, inputs = KiteModels.model!(kite, pos, vel)
outputs = []

println("get_control_function")
@time (f_ip, dvs, psym, kite_model) = get_control_function(kite_model, inputs; filename="kite_control_function.bin")
