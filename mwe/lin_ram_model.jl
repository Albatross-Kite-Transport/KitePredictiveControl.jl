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
# using ControlPlots

# TODO: THE VSM MODEL IS NOT HERE... FIX LINEARISATION OF THE VSM MODEL

toc()

# include(joinpath(@__DIR__, "plotting.jl"))

# Simulation parameters
dt = 0.005
total_time = 40.0  # Seconds of simulation after stabilization
vsm_interval = 0
steps = Int(round(total_time / dt))
trunc = false

# Initialize model
set_data_path(joinpath(dirname(@__DIR__), "data"))
set = load_settings("system_model.yaml")
set_values = [-60.0, 0.0, 0.0]  # Set values of the torques of the three winches. [Nm]
set.quasi_static = false
set.physical_model = "simple_ram"

@info "Creating RamAirKite model..."
s = RamAirKite(set)
toc()

measure = Measurement()

# Define outputs for linearization - angular velocities
@variables heading_x(t)
@variables winch_force(t)[1:3]
@variables tether_length(t)[1:3]

# Initialize at elevation with linearization outputs
s.point_system.winches[2].tether_length += 0.2
s.point_system.winches[3].tether_length += 0.2
measure.sphere_pos .= deg2rad.([75.0 75.0; 1.0 -1.0])
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
sys_state_op = KiteModels.SysState(s)
x_op = copy(s.integrator.u)

# Create loggers
logger_nonlinear = Logger(length(s.point_system.points), steps)
logger_linear = Logger(length(s.point_system.points), steps)  # Same structure for comparison

# Initialize system state trackers
sys_state_nonlinear = KiteModels.SysState(s)
sys_state_linear = deepcopy(sys_state_op) # Initialize linear state to operating point


function simulate(sys_state_op, sys_state_nonlinear, sys_state_linear)
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

    P = 1000.0
    setpoint = deg2rad(-0)
    lin_u = copy(u_op) .- 0.1
    x_lin = zeros(dsys.nx)
    sim_time = 0.0
    try
        for i in 1:steps
            if sim_time > 20
                setpoint = deg2rad(0.2)
            end
            push!(simulation_time_points, sim_time)
            # plot_kite(s, sim_time; zoom=false, front=false)

            # --- Calculate steering inputs ---
            nl_steering = P * (setpoint - s.integrator[sys.heading_x])
            nl_u = -s.set.drum_radius .* s.integrator[sys.winch_force] .+ [0, nl_steering, -nl_steering]

            # --- Nonlinear simulation step ---
            (t_new, _) = next_step!(s, nl_u; dt, vsm_interval=vsm_interval)
            sim_time = t_new - dt*stabilization_steps
            KiteModels.update_sys_state!(sys_state_nonlinear, s)
            sys_state_nonlinear.time = sim_time
            log!(logger_nonlinear, sys_state_nonlinear)

            # --- Linear simulation step ---
            # Create input matrix for lsim (single timestep)
            u_matrix = hcat(lin_u .- u_op, lin_u .- u_op)  # Create two identical input vectors
            t_vector = [0, dt] # Time vector for a single step

            # Simulate one step with lsim
            # --- Linearize at operating point ---
            # if i%20 == 0
            #     (; A, B, C, D) = KiteModels.linearize(s)
            #     csys = ss(A, B, C, D)
            #     dsys = c2d(csys, dt)
            #     if trunc
            #         dsys, hs, _ = baltrunc_unstab(dsys; residual=true, n=dsys.nx)
            #     else
            #         dsys = dsys
            #     end

            #     # Reset operating point
            #     sys_state_op = KiteModels.SysState(s)
            #     x_lin .= 0.0
            #     u_op .= s.get_set_values(s.integrator)
            # end

            u_pert = lin_u .- u_op  # Input perturbation
            xnext = dsys.A * x_lin .+ dsys.B * u_pert
            y_lin = dsys.C * xnext .+ dsys.D * u_pert
            x_lin .= xnext

            # Update linear system state
            sys_state_linear.heading = sys_state_op.heading + y_lin[1]
            sys_state_linear.force[1:3] .= sys_state_op.force[1:3] .+ y_lin[2:4]
            sys_state_linear.l_tether[1:3] .= sys_state_op.l_tether[1:3] .+ y_lin[5:7]
            lin_winch_force = sys_state_linear.force[1:3]
            lin_steering = P * (setpoint - sys_state_linear.heading)
            lin_u .= -s.set.drum_radius .* lin_winch_force .+ [0, lin_steering, -lin_steering]

            # Log the linear state
            sys_state_linear.time = sim_time
            log!(logger_linear, sys_state_linear)
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
    return simulation_time_points, steering_history, csys, dsys, tsys
end

simulation_time_points, steering_history, csys, dsys, tsys = simulate(sys_state_op, sys_state_nonlinear, sys_state_linear)

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
p3 = plot(t_common, [tether_length_nl[1], tether_length_lin[1]], title="Tether length [m]", label=["Nonlinear" "Linear"])
p4 = plot(t_common, [steering_history[1,:], steering_history[2,:]], title="Steering [Nm]", label=["Nonlinear" "Linear"])
bode = bodeplot([dsys[1,1], csys[1,1]])

p_comparison = plot(p1, p2, p3, p4, bode, layout=(3,2), size=(1200, 1200))
# p_comparison = plot(bode1, bode2, size=(1800, 1200))

display(p_comparison)

