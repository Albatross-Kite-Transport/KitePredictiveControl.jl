using Revise, KitePredictiveControl, KiteModels, ControlPlots

Ts = 0.05
if !@isdefined kite_real
    kite_real = KPS4_3L(KCU(se("system_3l.yaml")))
    kite_model = KPS4_3L(KCU(se("system_3l.yaml"))) # TODO: different sim vs real system
end

init_sim!(kite_real; prn=true, torque_control=false) # TODO: move to different core
init_sim!(kite_model; prn=true, torque_control=false)

ci = ControlInterface(kite_model; Ts)
init_sim!(kite_real; prn=true, torque_control=false)

try
    live_plot(ci)
    for i in 1:1000
        start_t = time()
        println("t = ", i*Ts-Ts)
        set_values = step!(ci, kite_real.integrator)
        next_step!(kite_real; set_values, dt = Ts)
        # plot2d(kite_real.pos, (i-1)*Ts; zoom=false, front=false, xlim=(-35, 35), ylim=(0, 70))
        pause_t = Ts - (time() - start_t)
        if (pause_t < 0) println("pause_t = ", pause_t)
        else sleep(pause_t) end
    end
finally
    sleep(2)
    stop_plot(ci)
end