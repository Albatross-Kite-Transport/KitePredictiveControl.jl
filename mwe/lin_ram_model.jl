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

using KiteModels, LinearAlgebra, Statistics, OrdinaryDiffEqCore 
using ModelingToolkit: t_nounits as t
using ModelingToolkit
using ControlSystems
using RobustAndOptimalControl
using Plots
# using ControlPlots

# TODO: figure out bodeplots https://juliacontrol.github.io/ControlSystemsMTK.jl/dev/batch_linearization/#Linearize-around-a-trajectory

toc()

# include(joinpath(@__DIR__, "plotting.jl"))

# Simulation parameters
dt = 0.05
total_time = 5.0  # Seconds of simulation after stabilization
vsm_interval = 3
steps = Int(round(total_time / dt))
trunc = true
steering_freq = 1/2  # Hz - full left-right cycle frequency
steering_magnitude = 1.0      # Magnitude of steering input [Nm]

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
@variables ω_b(t)[1:3]

# Initialize at elevation with linearization outputs
s.point_system.winches[2].tether_length += 0.2
s.point_system.winches[3].tether_length += 0.2
measure.sphere_pos .= deg2rad.([60.0 60.0; 1.0 -1.0])
KiteModels.init_sim!(s, measure; 
    remake=false,
    reload=true,
    lin_outputs=[ω_b[3]]  # Specify which outputs to track in linear model
)
sys = s.sys

@show rad2deg(s.integrator[sys.elevation])

@info "System initialized at:"
toc()

# --- Stabilize system at operating point ---
@info "Stabilizing system at operating point..."
s.integrator.ps[sys.stabilize] = true
stabilization_steps = Int(10 ÷ dt)
for i in 1:stabilization_steps
    next_step!(s; dt, vsm_interval=0.05÷dt)
end
s.integrator.ps[sys.stabilize] = false

# --- Get operating point values ---
# Extract state and input at operating point
u_op = copy(s.integrator[sys.set_values])
# Create a SysState to capture state values at operating point
sys_state_op = KiteModels.SysState(s)
# Also get the direct state vector for linear model
x_op = copy(s.integrator.u)

# Create loggers
logger_nonlinear = Logger(length(s.point_system.points), steps)
logger_linear = Logger(length(s.point_system.points), steps)  # Same structure for comparison

# Initialize system state trackers
sys_state_nonlinear = KiteModels.SysState(s)
sys_state_linear = deepcopy(sys_state_op) # Initialize linear state to operating point


function simulate(sys_state_op, sys_state_nonlinear, sys_state_linear)
    simulation_time_points = Float64[]
    steering_history = Float64[]
    perturbation_history = Vector{Float64}[]
    
    @info "Starting side-by-side simulation..."
    matrices = KiteModels.linearize(s)
    csys = ss(matrices...)
    dsys = c2d(csys, dt)
    if trunc
        tsys, hs, _ = baltrunc_unstab(dsys; residual=true, n=24)
        wanted_nx = count(x -> x > 1e-4, hs)
        @info "Nx should be $wanted_nx"
    else
        tsys = dsys
    end

    x_lin = zeros(tsys.nx)
    sim_time = 0.0
    try
        for i in 1:steps
            push!(simulation_time_points, sim_time)
            # plot_kite(s, sim_time; zoom=false, front=false)

            # --- Calculate steering inputs ---
            steering = steering_magnitude * cos(2π * steering_freq * sim_time)
            perturbation = [0.0, steering, -steering]

            u_op = -s.set.drum_radius .* s.integrator[sys.winch_force]
            s.lin_prob.ps[sys.set_values] = u_op

            set_values_nonlinear = u_op + perturbation
            push!(steering_history, steering)
            push!(perturbation_history, copy(perturbation))

            # --- Nonlinear simulation step ---
            (t_new, _) = next_step!(s, set_values_nonlinear; dt, vsm_interval=vsm_interval)
            sim_time = t_new - dt*stabilization_steps
            KiteModels.update_sys_state!(sys_state_nonlinear, s)
            sys_state_nonlinear.time = sim_time
            log!(logger_nonlinear, sys_state_nonlinear)

            # --- Linear simulation step ---
            # Create input matrix for lsim (single timestep)
            u_matrix = hcat(perturbation, perturbation)  # Create two identical input vectors
            t_vector = [0, dt] # Time vector for a single step

            # Simulate one step with lsim
            # --- Linearize at operating point ---
            if i%10 == 0
                (; A, B, C, D) = KiteModels.linearize(s)

                @show norm(A)
                csys = ss(A, B, C, D)
                dsys = c2d(csys, dt)
                if trunc
                    tsys, hs, _ = baltrunc_unstab(dsys; residual=true, n=tsys.nx)
                else
                    tsys = dsys
                end

                # Reset operating point
                sys_state_op = KiteModels.SysState(s)
                x_lin = zeros(tsys.nx) # Reset initial condition for lsim
            end

            y_lin, t_lin, x_lin_ = lsim(tsys, u_matrix, t_vector; x0=x_lin)
            x_lin .= x_lin_[:,end]

            # Update linear system state
            sys_state_linear.turn_rates = sys_state_op.turn_rates .+ y_lin[:,end] # Use the last output

            # Log the linear state
            sys_state_linear.time = sim_time
            log!(logger_linear, sys_state_linear)
        end
    catch e
        if isa(e, AssertionError)
            @show sim_time
            println(e)
        else
            rethrow(e)
        end
    end
    return simulation_time_points, steering_history, perturbation_history, csys, dsys, tsys
end

simulation_time_points, steering_history, perturbation_history, csys, dsys, tsys = simulate(sys_state_op, sys_state_nonlinear, sys_state_linear)

@info "Simulation completed"

# --- Save logs ---
save_log(logger_nonlinear, "nonlinear_model")
save_log(logger_linear, "linear_model")
lg_nonlinear = load_log("nonlinear_model")
lg_linear = load_log("linear_model")
sl_nonlinear = lg_nonlinear.syslog
sl_linear = lg_linear.syslog

# --- Plot comparison results ---
# Extract data
turn_rates_nl = rad2deg.(hcat(sl_nonlinear.turn_rates...))
turn_rates_lin = rad2deg.(hcat(sl_linear.turn_rates...))

# Find common time
t_common = sl_nonlinear.time

# Create comparison data
ω_x = [turn_rates_nl[1,:], turn_rates_lin[1,:]]
ω_y = [turn_rates_nl[2,:], turn_rates_lin[2,:]]
ω_z = [clamp.(turn_rates_nl[3,:], -2.5, 0.0), turn_rates_lin[3,:]]
input_series = [steering_history]

# Create Plots
p1 = plot(t_common, ω_x, title="ω_x [°/s]", label=["Nonlinear" "Linear (lsim)"])
p2 = plot(t_common, ω_y, title="ω_y [°/s]", label=["Nonlinear" "Linear (lsim)"])
p3 = plot(t_common, ω_z, title="ω_z [°/s]", label=["Nonlinear" "Linear (lsim)"])
p4 = plot(t_common, input_series, title="Steering [Nm]", label="Input")
bode1 = bodeplot(dsys)
bode2 = bodeplot(tsys)

# Combine plots into a single layout
p_comparison = plot(p1, p2, p3, p4, bode1, bode2, layout=(3,2), size=(1200, 1200))

display(p_comparison)

# --- Error metrics ---
error_ω_x = norm(ω_x[1,:] - ω_x[2,:]) / length(t_common)
error_ω_y = norm(ω_y[1,:] - ω_y[2,:]) / length(t_common)
error_ω_z = norm(ω_z[1,:] - ω_z[2,:]) / length(t_common)

@info "Error metrics (avg. L2 norm):" error_ω_x error_ω_y error_ω_z
