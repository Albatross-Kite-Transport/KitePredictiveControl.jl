# https://juliacontrol.github.io/ControlSystems.jl/stable/examples/automatic_differentiation/

using ControlSystemsBase, ForwardDiff, KiteModels, PreallocationTools, OrdinaryDiffEqCore, OrdinaryDiffEqBDF

Ts = 0.05
if !@isdefined kite
    kite = KPS4_3L(KCU(se("system_3l.yaml")))
end

init_set_values = [-0.1, -0.1, -70.0]
init_sim!(kite; prn=true, torque_control=true, init_set_values)
next_step!(kite; set_values = init_set_values, dt = 1.0)

sym_model, inputs = model!(kite, kite.pos, kite.vel)
@time KiteModels.ModelingToolkit.generate_control_function(sym_model, inputs)
@time KiteModels.ModelingToolkit.generate_control_function(sym_model, inputs)

solver = QNDF()
lbc = GeneralLazyBufferCache(function (x)
    println("creating prob")
    @time integ = OrdinaryDiffEqCore.init(ODEProblem(kite.simple_sys, x, (0.0, Ts)), solver; saveat=Ts, abstol=kite.set.abs_tol, reltol=kite.set.rel_tol)
    return integ
end)

"Nonlinear discrete dynamics"
function next_step(x, u, integrator)
    OrdinaryDiffEqCore.reinit!(integrator, x; t0=1.0, tf=1.0+Ts)
    solve!(integrator)
    return integrator.u
end
f = (x, u) -> next_step(x, u, lbc[x])


x0 = kite.integrator.u
u0 = init_set_values

println("linearizing")
@time A = ForwardDiff.jacobian(x -> f(x, u0), x0)
@time A = ForwardDiff.jacobian(x -> f(x, u0), x0)
@time B = ForwardDiff.jacobian(u -> f(x0, u), u0)

# "An example of a nonlinear output (measurement) function"
# function g(x, u)
#     y = [x[1] + 0.1x[1]*u[2]; x[2]]
# end

# C = ForwardDiff.jacobian(x -> g(x, u0), x0)
# D = ForwardDiff.jacobian(u -> g(x0, u), u0)

# linear_sys = ss(A, B, C, D)



# using Random, DifferentialEquations, LinearAlgebra, Optimization, OptimizationNLopt, OptimizationOptimJL, PreallocationTools

# lbc = GeneralLazyBufferCache(function (p)
#     DifferentialEquations.init(ODEProblem(ode_fnc, y₀, (0.0, T), p), Tsit5(); saveat = t)
# end)

# Random.seed!(2992999)
# λ, y₀, σ = -0.5, 15.0, 0.1
# T, n = 5.0, 200
# Δt = T / n
# t = [j * Δt for j in 0:n]
# y = y₀ * exp.(λ * t)
# yᵒ = y .+ [0.0, σ * randn(n)...]
# ode_fnc(u, p, t) = p * u
# function loglik(θ, data, integrator)
#     yᵒ, n, ε = data
#     λ, σ, u0 = θ
#     integrator.p = λ
#     reinit!(integrator, u0)
#     solve!(integrator)
#     ε = yᵒ .- integrator.sol.u
#     ℓ = -0.5n * log(2π * σ^2) - 0.5 / σ^2 * sum(ε.^2)
# end
# θ₀ = [-1.0, 0.5, 19.73]
# negloglik = (θ, p) -> -loglik(θ, p, lbc[θ[1]])
# fnc = OptimizationFunction(negloglik, Optimization.AutoForwardDiff())
# ε = zeros(n)
# prob = OptimizationProblem(fnc, θ₀, (yᵒ, n, ε), lb=[-10.0, 1e-6, 0.5], ub=[10.0, 10.0, 25.0])
# solve(prob, LBFGS())