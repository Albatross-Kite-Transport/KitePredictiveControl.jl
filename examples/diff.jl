# https://juliacontrol.github.io/ControlSystems.jl/stable/examples/automatic_differentiation/

using ControlSystemsBase, ForwardDiff, KiteModels, PreallocationTools, OrdinaryDiffEq, OrdinaryDiffEqCore, BenchmarkTools, ModelingToolkit
using ForwardDiff: Dual, JacobianConfig
using SymbolicIndexingInterface: parameter_values, state_values, setp, setu, getu
using SciMLStructures
import ModelingToolkit.SciMLBase: successful_retcode
using DifferentiationInterface
import ForwardDiff, PolyesterForwardDiff, FiniteDiff, FiniteDifferences, FastDifferentiation, Symbolics, Tracker

include("../src/mtk_interface.jl")
Ts = 0.05
if !@isdefined kite
    kite = KPS4_3L(KCU(se("system_3l.yaml")))
end

init_set_values = [-0.1, -0.1, -70.0]
init_sim!(kite; prn=true, torque_control=true, init_set_values, ϵ=100.0)
next_step!(kite; set_values = init_set_values, dt = 1.0)

sys = kite.prob.f.sys
inputs = [sys.set_values[i] for i in 1:3]
outputs = vcat(
    vcat(sys.flap_angle), 
    reduce(vcat, collect(sys.pos[:, 4:kite.num_flap_C-1])), 
    reduce(vcat, collect(sys.pos[:, kite.num_flap_D+1:kite.num_A])),
    vcat(sys.tether_length),
    sys.heading_y,
    sys.depower,
)

solver = OrdinaryDiffEq.QNDF()
(f!, h!, nu, nx, ny) = generate_f_h(kite, inputs, outputs, solver, Ts)

x0 = copy(kite.integrator.u)
u0 = copy(init_set_values)

xnext0 = zeros(nx)
f!(xnext0, x0, u0, 1.0, 1.0)

A = zeros(nx, nx)
B = zeros(nx, nu)
myf_x0!(xnext0, x0) = f!(xnext0, x0, u0, 1.0, 1.0)
myf_x0(x0) = f!(xnext0, x0, u0, 1.0, 1.0)
myf_u0!(xnext0, u0) = f!(xnext0, x0, u0, 1.0, 1.0)
myf_u0(u0) = f!(xnext0, x0, u0, 1.0, 1.0)

f_prob!(xnext0, x0, u0, t) = f!(xnext0, x0, u0, 1.0, 1.0)

# TODO: acc is pos-dependent, add acc to state variables.

function jac(backend)
    global A, B
    x0 = copy(kite.integrator.u)
    prep = prepare_jacobian(myf_x0!, xnext0, backend, x0)
    jacobian!(myf_x0!, xnext0, A, prep, backend, x0)
    @time jacobian!(myf_x0!, xnext0, A, prep, backend, x0)
    jacobian!(myf_u0!, xnext0, B, backend, u0)
    t = @elapsed jacobian!(myf_x0!, xnext0, A, prep, backend, x0)

    f!(xnext0, x0, u0, 1.0, 1.0)
    lin_xnext0 = A * x0 + B * u0
    diff_ulin = norm(xnext0 .- x0)
    diff_lin = norm(lin_xnext0 .- x0)
    println("solver: ", backend, "\n\tdiff_ulin: ", diff_ulin, "\n\tdiff_lin: ", diff_lin, "\n\ttime ", t)
    nothing
end

jac(AutoForwardDiff()) # works, big diff
# jac(AutoPolyesterForwardDiff()) # ERROR: LoadError: Cannot determine ordering of Dual tags ForwardDiff.Tag{DiffEqBase.OrdinaryDiffEqTag, Dual{Nothing, Float64, 11}} and Nothing
jac(AutoFiniteDiff()) # very fast

# FiniteDifferenceMethod(
#     grid::AbstractVector{Int},
#     q::Int;
#     condition::Real=DEFAULT_CONDITION,
#     factor::Real=DEFAULT_FACTOR,
#     max_range::Real=Inf
# )

function difference(points, order, adapt, factor, condition)
    # fdm = FiniteDifferences.FiniteDifferenceMethod([0, 1], 1);
    method = FiniteDifferences.forward_fdm(points, order; adapt, factor, condition);
    A, = FiniteDifferences.jacobian(method, myf_x0, x0);
    t = @elapsed A, = FiniteDifferences.jacobian(method, myf_x0, x0);
    B, = FiniteDifferences.jacobian(method, myf_u0, u0);
    f!(xnext0, x0, u0, 1.0, 1.0);
    lin_xnext0 = A * x0 + B * u0;
    diff_ulin = norm(xnext0 .- x0);
    diff_lin = norm(lin_xnext0 .- x0);
    println("solver: ", "FiniteDifferences", "\n\tdiff_ulin: ", diff_ulin, "\n\tdiff_lin: ", diff_lin, "\n\ttime ", t);
end

difference(2, 1, 2, 2, 10)

# jac(AutoFastDifferentiation()) ERROR: TypeError: non-boolean (FastDifferentiation.Node) used in boolean context
# jac(AutoSymbolics()) # slow
