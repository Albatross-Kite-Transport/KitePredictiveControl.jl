using Revise, KiteModels, LinearAlgebra, Statistics, KitePredictiveControl
using ControlPlots

set = deepcopy(load_settings("system_3l.yaml"))
# set.elevation = 71
dt = 0.01
total_time = 2.0

steps = Int(round(total_time / dt))
logger = Logger(3*set.segments + 6, steps)

if !@isdefined s; s = KPS4_3L(KCU(set)); end
s.set = update_settings()
s.set.abs_tol = 0.0006
s.set.rel_tol = 0.001
s.set.l_tether = 21.0
# s.set.damping = 473
s.set.elevation = 87
init_set_values = [-0.1, -0.1, -120.0]
@time KiteModels.init_sim!(s; prn=true, torque_control=true, init_set_values, ϵ=1e-3, flap_damping=0.1)
# @time next_step!(s; set_values=[0.0, 0.0, 0.0], dt=2.0)
println("vel ", mean(norm.(s.integrator[s.simple_sys.force])))
sys_state = KiteModels.SysState(s)

ci = KitePredictiveControl.ControlInterface(s; Ts=dt, u0=zeros(3), noise=0.0)

println("stepping")
total_step_time = 0.0
steering = zeros(3)
amount = 0.3
sign = 1
lin_state = ci.simple_h!(zeros(ci.linmodel.nx), s.integrator.u)
@show lin_state
for i in 1:steps
    global total_step_time, sys_state, steering, sign, lin_state
    time = (i-1) * dt
    Core.println("time: ", time)
    # if time > 1.0 steering = [-1, -20, -200.0] end
    # if time > 5.0 steering = [-10, -1, -200.0] end
    if time <= 1.0
        steering[2] = -3.0 
        steering[1] = 3.0 
        steering[3] = -0.0 
    end
    if time > 1.0
        steering[2] = 3.0 
        steering[1] = -3.0 
        steering[3] = -0.0 
    end

    # if time < 1.0
    #     steering[1] = 0.6
    #     steering[2] = -0.6
    # else
    #     steering .= 0.0
    # end
    lin_state .= (ci.linmodel.A * lin_state + ci.linmodel.Bu * steering)

    sys_state.var_01 =  rad2deg(s.get_flap_angle(s.integrator)[1])
    sys_state.var_02 =  rad2deg(s.get_flap_angle(s.integrator)[2])
    sys_state.var_03 =  rad2deg(s.integrator[s.simple_sys.depower])
    sys_state.var_04 =  s.tether_lengths[1]
    sys_state.var_05 =  s.tether_lengths[2]
    sys_state.var_06 =  lin_state[1]
    sys_state.var_07 =  s.integrator[s.simple_sys.turn_rate_y]
    sys_state.var_08 =  s.integrator[s.simple_sys.heading_y]
    sys_state.var_09 =  clamp(s.integrator[s.simple_sys.turn_rate_y] / (s.get_flap_angle(s.integrator)[2] - s.get_flap_angle(s.integrator)[1]), -10, 10)
    sys_state.var_10 =  (s.integrator[s.simple_sys.flap_angle][2] - s.integrator[s.simple_sys.flap_angle][1])
    sys_state.var_11 =  clamp((s.integrator[s.simple_sys.flap_vel][1] - s.integrator[s.simple_sys.flap_vel][2]) /
                            (steering[2] - steering[1]) * 1000, -10, 10)
    sys_state.var_12 =  lin_state[2]

    @show sys_state.var_08

    step_time = @elapsed next_step!(s; set_values=steering - winch_force(s)*0.11, dt=dt)
    if time > total_time/2
        total_step_time += step_time
    end
    s.t_0 = time

    KiteModels.update_sys_state!(sys_state, s)
    log!(logger, sys_state)
    l = s.set.l_tether+10
    # plot2d(s.pos, time; zoom=true, front=false, xlim=(-l/2, l/2), ylim=(0, l))
end

times_reltime = (total_time/2) / total_step_time
println("times realtime MTK model: ", times_reltime)
# println("avg steptime MTK model:   ", total_step_time/steps)

p=plotx(logger.time_vec, 
            [logger.var_01_vec,  logger.var_02_vec, logger.var_03_vec], 
            [logger.var_04_vec,  logger.var_05_vec], 
            [rad2deg.(logger.var_07_vec), rad2deg.(logger.var_08_vec), rad2deg.(logger.var_06_vec)], 
            [logger.var_09_vec, logger.var_11_vec],
            [logger.var_10_vec, logger.var_12_vec]; 
        ylabels=["steering", "length", "heading [deg]", "ratio", "distance"], 
        labels=[
            ["steering pos C", "steering pos D", "power angle"], 
            ["left tether", "right tether"], 
            ["turn_rate_y", "heading_y", "lin heading"],
            ["turn_rate / flap difference", "flap_diff' / steering_diff"],
            ["flap difference", "lin flap diff"]],
        fig="Steering and heading MTK model")
display(p)