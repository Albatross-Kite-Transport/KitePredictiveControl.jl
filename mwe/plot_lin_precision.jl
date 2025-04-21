
# Initialize states
setstate!(model, x0)
setstate!(plant, x0)

# Storage for states
nonlin_states = fill(NaN, nx, N)
plant_states = fill(NaN, nx, N)
lin_states = fill(NaN, nx, N)
times = zeros(N)
nonlin_states[:,1] .= x0
plant_states[:,1] .= x0
lin_states[:,1] .= x0
t = 0.0
times[1] = t

linmodel = ModelPredictiveControl.linearize(model; u=u0, x=x0)

# Simulation loop
for i in 2:N
    global linmodel, t, u
    # Calculate inputs
    u = u0 .+ 10.0
    
    # Update states
    nonlin_states[:,i] = updatestate!(model, u)
    plant_states[:,i] = updatestate!(plant, u)
    lin_states[:,i] = updatestate!(linmodel, u)

    p_model.sx .= p_model.get_sx(p_model.integ)
    KiteModels.linearize_vsm!(s_model)
    KiteModels.linearize_vsm!(s_plant)
    if i % 10 == 0
        linearize!(linmodel, model; u=u, x=nonlin_states[:,i])
    end
    
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
nonlin_plant_error = norm(nonlin_states - plant_states) / N
nonlin_lin_error = norm(nonlin_states - lin_states) / N
@info "Model Comparison Errors:" nonlin_plant_error nonlin_lin_error
