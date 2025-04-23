
function test_model(p)
    norms = Float64[]
    for x in [p.x0, p.x0 .+ 0.01]
        for u in [p.u0, p.u0 .+ 1.0]
            for _ in 1:2
                xnext = f(x, u, nothing, p)
                xnext = f(x, u, nothing, p)
                push!(norms, norm(xnext))
                ynext = h(xnext, nothing, p)
                # @info "x: $(norm(x)) u: $(norm(u)) xnext: $(norm(xnext)) ynext: $(norm(ynext))"
            end
        end
    end
    if length(unique(norms))*2 != length(norms) 
        display(norms)
        throw(ArgumentError("Different inputs/states should produce different outputs"))
    end
    nothing
end

function reset_p!(p)
    p.sx .= p.sx0
    p.set_x(p.integ, p.x0) # needs x0 for vsm linearization
    p.set_sx(p.integ, p.sx0)
    return nothing
end

function plot_lin_precision()
    # Initialize states
    reset_p!(p_model)
    reset_p!(p_plant)
    KiteModels.linearize_vsm!(p_model.s)
    KiteModels.linearize_vsm!(p_plant.s)

    setstate!(model, p_model.x0)
    setstate!(plant, p_plant.x0)
    
    # Storage for states
    nonlin_states = fill(NaN, model.nx, N)
    plant_states = fill(NaN, plant.nx, N)
    lin_states = fill(NaN, model.nx, N)
    times = zeros(N)
    nonlin_states[:,1] .= p_model.x0
    plant_states[:,1] .= p_plant.x0
    lin_states[:,1] .= p_model.x0
    t = 0.0
    times[1] = t
    
    linmodel = ModelPredictiveControl.linearize(model; u=p_model.u0, x=p_model.x0)

    # Simulation loop
    for i in 2:N
        # Calculate inputs
        u = p_model.u0 .+ 10.0
        
        # Update states
        nonlin_states[:,i] = updatestate!(model, u)
        plant_states[:,i] = updatestate!(plant, u)
        lin_states[:,i] = updatestate!(linmodel, u)

        p_model.sx .= p_model.get_sx(p_model.integ)
        p_plant.sx .= p_plant.get_sx(p_plant.integ)
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
        [nonlin_states[5,:], plant_states[9,:], lin_states[5,:]],
        [nonlin_states[7,:], plant_states[11,:], lin_states[7,:]],
        [nonlin_states[9,:], plant_states[13,:], lin_states[9,:]];
        ylabels=["Tether length 1", "Tether length 2", "Tether length 3"],
        labels=[
            ["Nonlin", "Plant", "Linear"],
            ["Nonlin", "Plant", "Linear"],
            ["Nonlin", "Plant", "Linear"]
        ],
        fig="Model Comparison")
    display(p)
end