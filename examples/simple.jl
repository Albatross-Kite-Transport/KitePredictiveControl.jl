using Revise, KitePredictiveControl, KiteModels

if !@isdefined kite
    Ts = 0.05
    kite = KPS4_3L(KCU(se("system_3l.yaml")))
    init_sim!(kite; torque_control=false)
end

control = ControlInterface(kite; Ts)
init_sim!(kite; prn=true, torque_control=false)

try
live_plot(control)
for i in 1:200
    @show i
    step!(control, kite.integrator)
    next_step!(kite; dt = Ts)
end
finally
sleep(2)
stop_plot(control)
end