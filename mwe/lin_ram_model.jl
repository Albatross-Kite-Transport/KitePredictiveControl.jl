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
using ControlPlots

toc()

include(joinpath(@__DIR__, "plotting.jl"))

# Simulation parameters
dt = 0.05
total_time = 1.0  # Seconds of simulation after stabilization
vsm_interval = 3
steps = Int(round(total_time / dt))

# Initialize model
set_data_path(joinpath(dirname(@__DIR__), "data"))
set = load_settings("system_model.yaml")
set_values = [-50.0, 0.0, 0.0]  # Set values of the torques of the three winches. [Nm]
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
measure.sphere_pos .= deg2rad.([70.0 70.0; 1.0 -1.0])
KiteModels.init_sim!(s, measure; 
    remake=false,
    reload=true,
    lin_outputs=[ω_b...]  # Specify which outputs to track in linear model
)
sys = s.sys

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

# --- Linearize at operating point ---
@info "Linearizing system at operating point..."
@time (; A, B, C, D) = KiteModels.linearize(s)
@info "System linearized with matrix dimensions:" A=size(A) B=size(B) C=size(C) D=size(D)
csys = ss(A, B, C, D)
@time dsys = c2d(csys, dt)
@time tsys, hs, _ = baltrunc_unstab(dsys; residual=true, n=23)

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

# --- Prepare the simulation ---
sim_time = 0.0
simulation_time_points = Float64[]
# Input history for plotting
input_history = Vector{Float64}[]
# Perturbation input history (will be used for lsim)
perturbation_history = Vector{Float64}[]

x_lin = zeros(tsys.nx)

@info "Starting side-by-side simulation..."
# Begin simulation
try
    sim_time = 0.0  # Start at t=0 for the comparison

    while sim_time < total_time
        global x_lin

        push!(simulation_time_points, sim_time)
        
        # --- Calculate steering inputs ---
        steering = 10.0
        perturbation = [0.0, steering, -steering]
        set_values_nonlinear = u_op .+ perturbation
        push!(input_history, copy(set_values_nonlinear))
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
        y_lin, t_lin, x_lin_ = lsim(tsys, u_matrix, t_vector; x0=x_lin)
        x_lin .= x_lin_[:,end]

        # Update linear system state
        @show y_lin[:,end]
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

@info "Nonlinear simulation completed"

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
steering = [inputs[2] - u_op[2] for inputs in input_history]

# Find common time
t_common = sl_nonlinear.time[1:min(length(sl_nonlinear.time), length(sl_linear.time))]
min_length = length(t_common)

# Trim data
turn_rates_nl = turn_rates_nl[:, 1:min_length]
turn_rates_lin = turn_rates_lin[:, 1:min_length]
steering = steering[1:min_length]

# Create comparison data
ω_x = [turn_rates_nl[1,:], turn_rates_lin[1,:]]
ω_y = [turn_rates_nl[2,:], turn_rates_lin[2,:]]
ω_z = [turn_rates_nl[3,:], turn_rates_lin[3,:]]
input_series = [steering]

# Plot
p_comparison = plotx(t_common,
    ω_x, ω_y, ω_z, input_series;
    ylabels=["ω_x [°/s]", "ω_y [°/s]", "ω_z [°/s]", "Steering [Nm]"],
    labels=[
        ["Nonlinear", "Linear (lsim)"],
        ["Nonlinear", "Linear (lsim)"],
        ["Nonlinear", "Linear (lsim)"],
        ["Input"]
    ],
    fig="Linear vs Nonlinear Model Comparison (lsim)")
display(p_comparison)

# --- Error metrics ---
error_ω_x = norm(ω_x[1] - ω_x[2]) / min_length
error_ω_y = norm(ω_y[1] - ω_y[2]) / min_length
error_ω_z = norm(ω_z[1] - ω_z[2]) / min_length

@info "Error metrics (avg. L2 norm):" error_ω_x error_ω_y error_ω_z