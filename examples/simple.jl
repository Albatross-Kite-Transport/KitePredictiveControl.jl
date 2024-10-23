using Revise, KitePredictiveControl, KiteModels, ControlPlots

Ts = 0.1
if !@isdefined kite_real
    kite_real = KPS4_3L(KCU(se("system_3l.yaml")))
    kite_model = KPS4_3L(KCU(se("system_3l.yaml"))) # TODO: different sim vs real system
end

init_set_values = [-0.1, -0.1, -70.0]
init_sim!(kite_real; prn=true, torque_control=true, init_set_values) # TODO: move to different core
init_sim!(kite_model; prn=true, torque_control=true, init_set_values)
next_step!(kite_real; set_values = init_set_values, dt = 1.0)
next_step!(kite_model; set_values = init_set_values, dt = 1.0)

ci = ControlInterface(kite_model; Ts, u0=init_set_values)
# init_sim!(kite_real; prn=true, torque_control=true, init_set_values=[-10, -10, -70])

function wanted_heading_y(t)
    wanted_heading_y = 0.0
    if t < 10.0
        wanted_heading_y = 0.0
    elseif t < 100.0
        wanted_heading_y = deg2rad(5)
    elseif t < 200.0
        wanted_heading_y = -deg2rad(5)
    else
        wanted_heading_y = deg2rad(5)
    end
    return wanted_heading_y
end

total_time = 30
try
    live_plot(ci)
    for i in 1:Int(div(total_time, Ts))
        start_t = time()
        t = i*Ts-Ts
        println("t = ", t)
        # ci.wanted_outputs[KitePredictiveControl.idx(ci.linmodel.yname, ci.sys.heading_y)] = wanted_heading_y(t)
        y = zeros(ci.nonlinmodel.ny)
        ci.nonlinmodel.h!(y, kite_real.integrator.u, nothing, nothing)
        set_values = KitePredictiveControl.step!(ci, y)
        real = @elapsed next_step!(kite_real; set_values, dt = Ts)
        # plot2d(kite_real.pos, (i-1)*Ts; zoom=false, front=false, xlim=(-35, 35), ylim=(0, 70))
        pause_t = Ts - (time() - start_t)
        if (pause_t + real < 0) println("pause_t = ", pause_t) end
        # elseif (pause_t > 0) sleep(pause_t) end
    end
finally
    sleep(2)
    stop_plot(ci)
end