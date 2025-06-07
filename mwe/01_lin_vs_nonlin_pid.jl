# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

#=
This example demonstrates linearized model accuracy by comparing:
1. Nonlinear RamAirKite model simulation 
2. Linearized state-space model simulation, one timestep at a time within the loop

Both models start from the same operating point and are subjected
to identical steering inputs. The resulting state trajectories are
plotted together to visualize how well the linearized model
approximates the nonlinear dynamics.

This version uses lsim from ControlSystems.jl for the linear simulation.
=#

using Timers
tic()
@info "Loading packages "

using KiteModels, LinearAlgebra, Statistics, OrdinaryDiffEq
using KiteModels: linearize
using ModelingToolkit: t_nounits as t
using ModelingToolkit
using ControlSystems
using RobustAndOptimalControl
using Plots
using DiscretePIDs
# using ControlPlots

# TODO: THE VSM MODEL IS NOT HERE... FIX LINEARISATION OF THE VSM MODEL

toc()

# include(joinpath(@__DIR__, "plotting.jl"))

# Simulation parameters
dt = 0.05
total_time = 100.0  # Seconds of simulation after stabilization
vsm_interval = 0
steps = Int(round(total_time / dt))
trunc = false

# Initialize model
set_data_path(joinpath(dirname(@__DIR__), "data"))
set = load_settings("system_model.yaml")
set_values = [-60.0, 0.0, 0.0]  # Set values of the torques of the three winches. [Nm]
set.quasi_static = true
set.physical_model = "ram"

@info "Creating RamAirKite model..."
s = RamAirKite(set)
toc()

measure = Measurement()

# Define outputs for linearization - angular velocities
@variables heading_x(t)
@variables winch_force(t)[1:3]
@variables tether_length(t)[1:3]

# Initialize at elevation with linearization outputs
s.point_system.winches[2].tether_length -= 0.2
s.point_system.winches[3].tether_length -= 0.2
measure.sphere_pos .= deg2rad.([70.0 70.0; 1.0 -1.0])
KiteModels.init_sim!(s, measure; 
    remake=false,
    reload=true,
    lin_outputs=[heading_x, winch_force..., tether_length...]  # Specify which outputs to track in linear model
)
sys = s.sys

@show rad2deg(s.integrator[sys.elevation])

@info "System initialized at:"
toc()

# --- Stabilize system at operating point ---
@info "Stabilizing system at operating point..."
s.integrator.ps[sys.stabilize] = true
stabilization_steps = Int(10 ÷ dt)
# stabilization_steps = 0
for i in 1:stabilization_steps
    next_step!(s; dt, vsm_interval=0.05÷dt)
end
s.integrator.ps[sys.stabilize] = false
# for i in 1:stabilization_steps÷2
#     nl_steering = P * (0 - s.integrator[sys.heading_x])
#     nl_u = -s.set.drum_radius .* s.integrator[sys.winch_force] .+ [0, -nl_steering, nl_steering]
#     (t_new, _) = next_step!(s, nl_u; dt, vsm_interval=vsm_interval)
# end

# --- Get operating point values ---
u_op = copy(-s.set.drum_radius .* s.integrator[sys.winch_force])
s.set_set_values(s.integrator, u_op)
ss_op = KiteModels.SysState(s)
x_op = copy(s.integrator.u)

# Create loggers
logger_nonlinear = Logger(length(s.point_system.points), steps)
logger_linear = Logger(length(s.point_system.points), steps)  # Same structure for comparison

# Initialize system state trackers
ss_nl = KiteModels.SysState(s)
ss_lin = KiteModels.SysState(s)


function simulate(ss_op, ss_nl, ss_lin)
    simulation_time_points = Float64[]
    steering_history = fill(NaN, 2, steps)
    
    @info "Starting side-by-side simulation..."
    matrices = KiteModels.linearize(s)
    
    csys = ss(matrices...)
    if trunc
        tsys, hs, _ = baltrunc_unstab(csys; residual=true, n=26)
        wanted_nx = count(x -> x > 1e-4, hs)
        @info "Nx should be $wanted_nx"
    else
        tsys = csys
    end
    dsys = c2d(tsys, dt)

    nl_steering_pid, lin_steering_pid = [DiscretePID(; K=10, Ti=5, Ts=dt) for _ in 1:2]
    nl_power_pid, lin_power_pid = [DiscretePID(; K=100, Ti=10, Ts=dt) for _ in 1:2]
    nl_line_pid, lin_line_pid = [DiscretePID(; K=100, Ti=10, Ts=dt) for _ in 1:2]
    setpoint = deg2rad(-0)
    
    nl_u = copy(u_op)
    lin_u = copy(u_op) # Start with operating point input

    x_lin = zeros(dsys.nx) # State deviation for the discrete linear system
    sim_time = 0.0
    try
        for i in 1:steps
            if i == steps ÷ 2
                setpoint = deg2rad(5)
                csys = ss(matrices...)
                if trunc
                    tsys, hs, _ = baltrunc_unstab(csys; residual=true, n=26)
                    wanted_nx = count(x -> x > 1e-4, hs)
                    @info "Nx should be $wanted_nx"
                else
                    tsys = csys
                end
                dsys = c2d(tsys, dt)
                lin_u = copy(u_op) # Start with operating point input
                x_lin = zeros(dsys.nx) # State deviation for the discrete linear system
                ss_op = SysState(s)
                ss_lin = SysState(s)
                nl_steering_pid.I = 0.0
                lin_steering_pid.I = 0.0
                nl_power_pid.I = 0.0
                lin_power_pid.I = 0.0
                nl_line_pid.I = 0.0
                lin_line_pid.I = 0.0
            end
            push!(simulation_time_points, sim_time)

            # --- Nonlinear simulation step ---
            (t_new, _) = next_step!(s, nl_u; dt, vsm_interval=vsm_interval)
            if i == 1
                sim_time = dt # First step after stabilization
            else
                sim_time += dt
            end
            KiteModels.update_sys_state!(ss_nl, s)
            ss_nl.time = sim_time
            log!(logger_nonlinear, ss_nl)
            # --- Calculate steering inputs ---
            # Nonlinear controller
            mean_op_length = (ss_op.l_tether[2] + ss_op.l_tether[3]) / 2
            mean_nl_length = (ss_nl.l_tether[2] + ss_nl.l_tether[3]) / 2
            nl_steering = nl_steering_pid(setpoint, ss_nl.heading)
            nl_line = nl_line_pid(mean_op_length, mean_nl_length)
            nl_u .= [
                nl_power_pid(ss_op.l_tether[1], ss_nl.l_tether[1]),
                nl_line + nl_steering,
                nl_line - nl_steering
            ]

            # --- Linear simulation step ---
            u_pert = lin_u .- u_op  # Input perturbation for the discrete linear system
            xnext = dsys.A * x_lin .+ dsys.B * u_pert
            y_lin_dev = dsys.C * xnext .+ dsys.D * u_pert # y_lin_dev is the deviation of outputs
            x_lin .= xnext
            # Update absolute linear system state from deviations
            ss_lin.heading = ss_op.heading + y_lin_dev[1]
            ss_lin.force[1:3] .= ss_op.force[1:3] .+ y_lin_dev[2:4]
            ss_lin.l_tether[1:3] .= ss_op.l_tether[1:3] .+ y_lin_dev[5:7]
            mean_lin_length = (ss_lin.l_tether[2] + ss_lin.l_tether[3]) / 2
            lin_line = lin_line_pid(mean_op_length, mean_lin_length)
            lin_steering = lin_steering_pid(setpoint, ss_lin.heading)
            lin_u .= [
                lin_power_pid(ss_op.l_tether[1], ss_lin.l_tether[1]),
                lin_line + lin_steering,
                lin_line - lin_steering
            ]
            # Log the linear state
            ss_lin.time = sim_time # Use the same sim_time
            log!(logger_linear, ss_lin)
            steering_history[:, i] .= [nl_steering, lin_steering]
        end
    catch e
        if isa(e, AssertionError)
            @show sim_time
            println(e)
        else
            rethrow(e)
        end
    end
    return simulation_time_points, steering_history, csys, dsys, tsys, nl_power_pid
end

simulation_time_points, steering_history, csys, dsys, tsys, nl_power_pid = simulate(ss_op, ss_nl, ss_lin)

@info "Simulation completed"

# --- Save logs ---
save_log(logger_nonlinear, "nonlinear_model")
save_log(logger_linear, "linear_model")
lg_nonlinear = load_log("nonlinear_model")
lg_linear = load_log("linear_model")
sl_nl = lg_nonlinear.syslog
sl_lin = lg_linear.syslog

# --- Plot comparison results ---
heading_nl = rad2deg.(sl_nl.heading)
heading_lin = rad2deg.(sl_lin.heading)
winch_force_nl = [rad2deg(f[1]) for f in sl_nl.force]
winch_force_lin = [rad2deg(f[1]) for f in sl_lin.force]
tether_length_nl = [[l[i] for l in sl_nl.l_tether] for i in 1:3]
tether_length_lin = [[l[i] for l in sl_lin.l_tether] for i in 1:3]

t_common = sl_nl.time

# Create Plots
p1 = plot(t_common, [heading_nl, heading_lin], title="heading [°]", label=["Nonlinear" "Linear"])
p2 = plot(t_common, [winch_force_nl, winch_force_lin], title="Winch force [N]", label=["Nonlinear" "Linear"])
p3 = plot(t_common, [tether_length_nl[1], tether_length_lin[1], tether_length_nl[2], tether_length_lin[2]], 
            title="Tether length [m]", label=["Nonlinear power" "Linear power" "Nonlinear steering" "Linear steering"])
p4 = plot(t_common, [steering_history[1,:], steering_history[2,:]], title="Steering [Nm]", label=["Nonlinear" "Linear"])
bode = bodeplot([dsys[2,2], csys[2,2]], label=["discrete sys" "continuous sys"])

p_comparison = plot(p1, p2, p3, p4, bode, layout=(3,2), size=(1200, 1200))
# p_comparison = plot(bode1, bode2, size=(1800, 1200))

display(p_comparison)

