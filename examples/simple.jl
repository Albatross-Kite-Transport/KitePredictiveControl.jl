using Revise, KitePredictiveControl, KiteModels, ControlPlots

Ts = 0.1
if !@isdefined kite_real
    kite_real = KPS4_3L(KCU(se("system_real.yaml")))
    kite_model = KPS4_3L(KCU(se("system_model.yaml"))) # TODO: different sim vs real system
end
kite_real.set = update_settings()
kite_model.set = update_settings()
kite_real.set.l_tether = 50.0
kite_model.set.l_tether = 50.0
kite_real.set.elevation = 87
kite_model.set.elevation = 87
init_set_values = [-0.1, -0.1, -120.0]
init_sim!(kite_real; prn=true, torque_control=true, init_set_values, ϵ=1e-3, flap_damping=0.1) # TODO: move to different core
init_sim!(kite_model; prn=true, torque_control=true, init_set_values, ϵ=1e-3, flap_damping=0.1)

ci = KitePredictiveControl.ControlInterface(kite_model; Ts=Ts, u0=zeros(3), noise=0.0)

# init_sim!(kite_real; prn=true, torque_control=true, init_set_values=[-10, -10, -70])

# @assert false
# wanted_heading = deg2rad(1)
total_time = 100
try
    # start_processes!(ci)
    for i in 1:Int(div(total_time, Ts))
        # global wanted_heading
        start_t = time()
        t = i*Ts-Ts
        println("t = ", t)

        y = zeros(ci.linmodel.ny)
        ci.simple_h!(y, kite_real.integrator.u)
        @show kite_real.pos[kite_real.num_A][2] 
        # if (kite_real.pos[kite_real.num_A][2] < -3.0) wanted_heading = -deg2rad(1.0) end
        # if (kite_real.pos[kite_real.num_A][2] > 3.0) wanted_heading = deg2rad(1.0) end
        wanted_heading = deg2rad(sin(2π * t / 20) * (5 + t/total_time * 10))
        set_values = KitePredictiveControl.step!(ci, kite_real.integrator.u, y; rheading=wanted_heading)
        real = @elapsed next_step!(kite_real; set_values = set_values .- winch_force(kite_real) * kite_real.set.drum_radius, dt = Ts)
        # plot2d(kite_real.pos, (i-1)*Ts; zoom=false, front=false, xlim=(-35, 35), ylim=(0, 70))
        pause_t = Ts - (time() - start_t)
        println("Run percentage: ", (time() - start_t)/Ts*100)
        if (pause_t + real < 0) println("pause_t = ", pause_t)
        elseif (pause_t > 0) sleep(pause_t) end
        l = kite_real.set.l_tether+10
        plot2d(kite_real.pos, (i-1)*Ts; zoom=true, front=true, xlim=(-l/2, l/2), ylim=(0, l))
    end
finally
    # sleep(2)
    # stop_processes!(ci)
end