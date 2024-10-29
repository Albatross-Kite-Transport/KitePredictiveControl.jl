using Revise, KitePredictiveControl, KiteModels, ControlPlots

Ts = 0.05
if !@isdefined kite_real
    kite_real = KPS4_3L(KCU(se("system_3l.yaml")))
    kite_model = KPS4_3L(KCU(se("system_3l.yaml"))) # TODO: different sim vs real system
end

init_set_values = [-0.1, -0.1, -70.0]
init_sim!(kite_real; prn=true, torque_control=true, init_set_values) # TODO: move to different core
init_sim!(kite_model; prn=true, torque_control=true, init_set_values)
next_step!(kite_real; set_values = init_set_values, dt = 1.0)
next_step!(kite_model; set_values = init_set_values, dt = 1.0)

# if !@isdefined(ci)
    ci = KitePredictiveControl.ControlInterface(kite_model; Ts=Ts, u0=init_set_values, noise=0.0)
# else
#     reset!(ci; u0=init_set_values)
# end

# init_sim!(kite_real; prn=true, torque_control=true, init_set_values=[-10, -10, -70])

function wanted_heading_y(t)
    wanted_heading_y = 0.0
    if t < 10.0
        wanted_heading_y = 0.0
    elseif t < 100.0
        wanted_heading_y = deg2rad(20)
    elseif t < 200.0
        wanted_heading_y = -deg2rad(-1)
    else
        wanted_heading_y = deg2rad(5)
    end
    return wanted_heading_y
end

total_time = 40
try
    start_processes!(ci)
    for i in 1:Int(div(total_time, Ts))
        start_t = time()
        t = i*Ts-Ts
        println("t = ", t)

        y = zeros(ci.linmodel.ny)
        ci.h!(y, kite_real.integrator.u)
        set_values = KitePredictiveControl.step!(ci, kite_real.integrator.u, y; rheading=wanted_heading_y(t))
        real = @elapsed next_step!(kite_real; set_values, dt = Ts)
        # plot2d(kite_real.pos, (i-1)*Ts; zoom=false, front=false, xlim=(-35, 35), ylim=(0, 70))
        pause_t = Ts - (time() - start_t)
        println("Run percentage: ", (time() - start_t)/Ts*100)
        if (pause_t + real < 0) println("pause_t = ", pause_t)
        elseif (pause_t > 0) sleep(pause_t) end
    end
finally
    sleep(2)
    stop_processes!(ci)
end