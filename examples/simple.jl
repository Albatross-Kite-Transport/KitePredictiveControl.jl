using Revise, KitePredictiveControl, ControlPlots, KiteModels, KiteViewers

# --- user constants ---
plt = 2 # 0 = graph, 1 = 2d, 2 = 3d
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
if (plt==2) viewer::Viewer3D = Viewer3D(true) end
ss::SysState = SysState(kite_real, 1.0)

ci = KitePredictiveControl.ControlInterface(kite_model; Ts=Ts, u0=zeros(3), noise=0.0)

# --- heading ---
last_wanted_heading = deg2rad(30.0)
function wanted_heading(t, pos_y, last_wanted_heading)
    wanted_heading = last_wanted_heading
    if last_wanted_heading > 0 && pos_y < -20
        wanted_heading = -last_wanted_heading - deg2rad(0)
        last_wanted_heading = wanted_heading
    elseif last_wanted_heading < 0 && pos_y > 20
        wanted_heading = -last_wanted_heading + deg2rad(0)
        last_wanted_heading = wanted_heading
    end
    @show rad2deg(wanted_heading)
    return wanted_heading, last_wanted_heading
end

total_time = 60
try
    if (plt==0) start_processes!(ci) end
    for i in 1:Int(div(total_time, Ts))
        global last_wanted_heading
        start_t = time()
        t = i*Ts-Ts
        println("t = ", t)

        y = zeros(ci.model.ny)
        ci.h!(y, kite_real.integrator.u, nothing, nothing)
        pos_y = kite_real.pos[kite_real.num_A][2]
        @show pos_y
        # if (kite_real.pos[kite_real.num_A][2] < -3.0) wanted_heading = -deg2rad(1.0) end
        # if (kite_real.pos[kite_real.num_A][2] > 3.0) wanted_heading = deg2rad(1.0) end

        rheading, last_wanted_heading = wanted_heading(t, pos_y, last_wanted_heading)
        # rheading = deg2rad(-90)
        set_values = KitePredictiveControl.step!(ci, kite_real.integrator.u, y; rheading)
        update_sys_state!(ss, kite_real, 1.0)
        if (plt==2) update_system(viewer, ss, scale=1.0, kite_scale=1.0) end

        real = @elapsed next_step!(kite_real; set_values = set_values .- winch_force(kite_real) * kite_real.set.drum_radius, dt = Ts)
        pause_t = Ts - (time() - start_t)
        println("Run percentage: ", (time() - start_t)/Ts*100)
        if (pause_t + real < 0) println("pause_t = ", pause_t)
        elseif (pause_t > 0) sleep(pause_t) end
        l = kite_real.set.l_tether+10
        if (plt==1) plot2d(kite_real.pos, (i-1)*Ts; zoom=true, front=true, xlim=(-l, l), ylim=(0, 2l)) end
    end
finally
    if (plt==0) stop_processes!(ci) end
end