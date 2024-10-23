# https://juliacontrol.github.io/ControlSystems.jl/stable/examples/automatic_differentiation/

using ControlSystemsBase, ForwardDiff, KiteModels, PreallocationTools, OrdinaryDiffEqCore, OrdinaryDiffEqBDF, BenchmarkTools, ModelingToolkit
using ForwardDiff: Dual, JacobianConfig
using SymbolicIndexingInterface: parameter_values, state_values, setp, setu
using SciMLStructures

Ts = 0.05
if !@isdefined kite
    kite = KPS4_3L(KCU(se("system_3l.yaml")))
end

init_set_values = [-0.1, -0.1, -70.0]
init_sim!(kite; prn=true, torque_control=true, init_set_values)
# next_step!(kite; set_values = init_set_values, dt = 1.0)

x0 = copy(kite.integrator.u)
x = x0
u0 = init_set_values
nx = length(x0)
nu = length(u0)
sys = kite.prob.f.sys

idxs = [ModelingToolkit.parameter_index(kite.prob, sys.set_values[i]).idx for i in 1:3]

solver = QNDF()
# if !@isdefined(integrator)
    integ_cache = GeneralLazyBufferCache(
        function (xu)
            @show xu[1]
            @show xu[nx+1]
            par = vcat([sys.set_values[i] => xu[nx+i] for i in 1:3])
            # @show par
            x = xu[1:nx]
            u = xu[nx+1:end]
            default = create_default(x)
            prob = ODEProblem(sys, default, (0.0, Ts), par)
            setu! = setp(prob, [sys.set_values[i] for i in 1:3])
            integrator = OrdinaryDiffEqCore.init(prob, solver; saveat=Ts, abstol=kite.set.abs_tol, reltol=kite.set.rel_tol, verbose=false)
            # idxs = [parameter_index(integrator.f.sys, integrator.f.sys.set_values[i]) for i in 1:nu]
            # integrator.ps[idxs]
            # integrator.ps[idxs] .= xu[nx+1:end]
            # setx! = setu(integrator)
            return (integrator, setu!)
        end
    )
# end

function make_default_creator(sys)
    keys = collect(unknowns(sys))
    return x -> Dict(k => x[i] for (i, k) in enumerate(keys))
end

create_default = make_default_creator(sys)
@time default = create_default(x)


# setu! = setp(sys, [sys.set_values[i] for i in 1:3])
"Nonlinear discrete dynamics"
function next_step(x, u, integ_setu_pair)
    (integ, setu!) = integ_setu_pair
    reinit!(integ, x)
    setu!(integ, u)
    OrdinaryDiffEqCore.step!(integ, Ts, true)
    @assert successful_retcode(integ.sol)
    return integ.u
end
f(x, u) = next_step(x, u, integ_cache[vcat(x, u)])

# x = Dual{ForwardDiff.Tag{var"#173#174", Float64}, Float64, 11}[Dual{ForwardDiff.Tag{var"#173#174", Float64}}(12.42190583827353,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.41570907026443815,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(26.024821026029812,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(12.42190583823696,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(-0.41570907027871234,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(26.024821026047054,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(1.716459793066168,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(2.8656314901062423e-11,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(24.88602111894572,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.2713700064520312,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.27137000636095476,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(3.2872397842769754,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(5.839149244278587e-11,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(49.78170339502708,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(3.4836857908233805,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.7979792878868877,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(53.61499454158922,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(3.4836857907853878,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(-0.7979792877899043,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(53.61499454158683,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(4.056031513861585,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(3.492044624477669e-11,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(53.58054065769891,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(-1.2465326873313742,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(-0.020816073844370377,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.6000933142603445,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(-1.246532687282233,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.020816073850808494,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.6000933143460707,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(-0.8499666184280955,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(3.3093019640867786e-11,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.23243037004931996,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.15009401554829158,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.15009401561228283,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(-1.6528609290783103,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(1.9204037654598971e-10,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.4567823254980781,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(-1.7714260306222445,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.00011821769339786764,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.46442054012920525,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(-1.771426030952434,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(-0.00011821668523791292,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.46442054027495616,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(-1.7707124225753623,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(3.819544456761046e-10,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.4807101761277947,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(57.67693861558659,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(57.676938615586934,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(49.498364911999346,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.008946836579532528,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.008946836576655254,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{var"#173#174", Float64}}(0.2234100496025047,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)]
# u = [-0.1, -0.1, -70.0]

println("linearizing")
# # if !@isdefined(cfg_a)
#     f_a = x -> f(x, u0)
#     cfg_a = JacobianConfig(f_a, x0)
# # end
# # if !@isdefined(cfg_b)
#     f_b = u -> f(x0, u)
#     cfg_b = JacobianConfig(f_b, u0)
# # end
println("A")
@time A = ForwardDiff.jacobian(f_a, x0)
println("A")
@time A = ForwardDiff.jacobian(f_a, x0)
println("B")
@time B = ForwardDiff.jacobian(f_b, u0)
println("B")
@time B = ForwardDiff.jacobian(f_b, u0)

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