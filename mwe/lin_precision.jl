# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

using KiteModels, LinearAlgebra, ModelPredictiveControl, ModelingToolkit, OrdinaryDiffEq, FiniteDiff, DifferentiationInterface
using ModelingToolkit: setu, setp, getu, getp
using LaTeXStrings, ControlPlots, KiteUtils

ad_type = AutoFiniteDiff()

# Simulation parameters
dt = 0.05
total_time = 1.0
steps = Int(round(total_time / dt))

# Initialize model
set_data_path(joinpath(@__DIR__, "../data"))
set_model = deepcopy(se("system_model.yaml"))
dt = 1/set_model.sample_freq
s = RamAirKite(set_model)

set_plant = deepcopy(se("system_plant.yaml"))
s_plant = RamAirKite(set_plant)

# Create models (nonlinear and plant)
measure = Measurement()
measure.set_values .= [-55, -4.0, -4.0]
measure.sphere_pos .= deg2rad.([83.0 83.0; 1.0 -1.0])

KiteModels.init_sim!(s, measure; remake=false)
KiteModels.init_sim!(s_plant, measure; remake=false)

sys = s.sys

# Setup model parameters
function get_p(s)
    x_vec = KiteModels.get_unknowns(s)
    y_vec = [x_vec; s.sys.angle_of_attack; s.sys.heading_x; s.sys.tether_acc]
    inputs = collect(s.sys.set_values)
    
    set_x = setu(s.integrator, Initial.(x_vec))
    set_xh = setu(s.integrator, x_vec)
    set_u = setu(s.integrator, inputs)
    get_x = getu(s.integrator, x_vec)
    get_y = getu(s.integrator, y_vec)
    
    x0 = get_x(s.integrator)
    u0 = -s.set.drum_radius * s.integrator[sys.winch_force]
    
    return (s, set_x, set_u, get_x, get_y, dt, set_xh), x_vec, y_vec, x0, u0, inputs
end

# Model functions
solver = FBDF(nlsolve=OrdinaryDiffEq.NLNewton(relax=0.4))
function f(x, u, _, p)
    (s, set_x, set_u, get_x, _, _, _) = p
    set_x(s.prob, x)
    set_u(s.prob, u)
    sol = solve(s.prob, solver; dt, abstol=s.set.abs_tol, reltol=s.set.rel_tol, save_on=false, save_everystep=false, save_start=false, verbose=false)
    return get_x(sol)[1]
end

function h(x, _, p)
    (s, _, _, _, get_y, _, set_xh) = p
    set_xh(s.integrator, x)
    y = get_y(s.integrator)
    return y
end

# Setup models
p_model, x_vec, y_vec, x0, u0, inputs = get_p(s)
p_plant, _, _, _, _, _ = get_p(s_plant)

nu, nx, ny = length(inputs), length(x_vec), length(y_vec)
model = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_model, solver=nothing, jacobian=ad_type); 
                u=string.(inputs), x=string.(x_vec), y=string.(y_vec))
plant = setname!(NonLinModel(f, h, dt, nu, nx, ny; p=p_plant, solver=nothing, jacobian=ad_type);
                u=string.(inputs), x=string.(x_vec), y=string.(y_vec))

# Initialize states
setstate!(model, x0)
setstate!(plant, x0)

# Storage for states
nonlin_states = zeros(nx, steps)
plant_states = zeros(nx, steps)
lin_states = fill(NaN, nx, steps)
times = zeros(steps)
nonlin_states[:,1] .= x0
plant_states[:,1] .= x0
lin_states[:,1] .= x0
t = 0.0
times[1] = t

linmodel = ModelPredictiveControl.linearize(model; u=u0, x=x0)

# Simulation loop
for i in 2:steps
    global linmodel, t
    # Calculate inputs
    u = [-20, -1, -1]
    
    # Update states
    nonlin_states[:,i] = updatestate!(model, u)
    plant_states[:,i] = updatestate!(plant, u)
    lin_states[:,i] = updatestate!(linmodel, u)
    linearize!(linmodel, model; u=u, x=lin_states[:,i-1])
    
    t += dt
    times[i] = t
end

# Plot results
p = plotx(times, 
    [nonlin_states[5,:], plant_states[5,:], lin_states[5,:]],
    [nonlin_states[7,:], plant_states[7,:], lin_states[7,:]],
    [nonlin_states[9,:], plant_states[9,:], lin_states[9,:]];
    ylabels=["Tether length 1", "Tether length 2", "Tether length 3"],
    labels=[
        ["Nonlin", "Plant", "Linear"],
        ["Nonlin", "Plant", "Linear"],
        ["Nonlin", "Plant", "Linear"]
    ],
    fig="Model Comparison")
display(p)

# Print error metrics
nonlin_plant_error = norm(nonlin_states - plant_states) / steps
nonlin_lin_error = norm(nonlin_states - lin_states) / steps
@info "Model Comparison Errors:" nonlin_plant_error nonlin_lin_error
