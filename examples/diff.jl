# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

# https://juliacontrol.github.io/ControlSystems.jl/stable/examples/automatic_differentiation/

using ControlSystemsBase, ForwardDiff, KiteModels, PreallocationTools, OrdinaryDiffEq, OrdinaryDiffEqCore, BenchmarkTools, ModelingToolkit
using ForwardDiff: Dual, JacobianConfig
using SymbolicIndexingInterface: parameter_values, state_values, setp, setu, getu
using SciMLStructures
import ModelingToolkit.SciMLBase: successful_retcode
using SparseDiffTools
using SparseArrays
using LinearAlgebra

include("../src/mtk_interface.jl")
Ts = 0.1
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
(f!, h!, idxs, nu, nx, nx_simple, ny) = generate_f_h(kite, inputs, outputs, solver, Ts)
state = unknowns(sys)[idxs]

x0 = copy(kite.integrator.u)
simple_x0 = x0[idxs]
u0 = copy(init_set_values)

xnext0 = zeros(nx_simple)
f!(xnext0, x0, simple_x0, u0)

A = zeros(nx_simple, nx_simple)
B = zeros(nx_simple, nu)
myf_x0!(xnext0, simple_x0) = f!(xnext0, x0, simple_x0, u0)
myf_u0!(xnext0, u0) = f!(xnext0, x0, simple_x0, u0)

function idx(symbol)
    return ModelingToolkit.variable_index(string.(state), symbol)
end

"""
pos' = vel
vel' = acc
acc' = k_1 + k_2 * set_value
k_1' = 0
"""
function jac(backend)
    jac_prototype = ones(nx_simple, nx_simple)
    # [jac_prototype[idx(sys.tether_length[i]), idx(sys.tether_vel[i])] = 1.0 for i in 1:3]
    # [jac_prototype[idx(sys.tether_length[i]), idx(sys.tether_length[i])] = 1.0 for i in 1:3]
    sd = JacPrototypeSparsityDetection(; jac_prototype=sparse(jac_prototype))
    adtype = AutoSparse(AutoForwardDiff())

    cache_A = sparse_jacobian_cache(adtype, sd, myf_x0!, xnext0, simple_x0)
    A = sparse_jacobian(adtype, cache_A, myf_x0!, xnext0, simple_x0)
    t = @elapsed sparse_jacobian!(A, adtype, cache_A, myf_x0!, xnext0, simple_x0)


    jac_prototype = ones(nx_simple, nu)
    # jac_prototype[vel_idxs, :] .= 1.0
    sd = JacPrototypeSparsityDetection(; jac_prototype=sparse(jac_prototype))
    adtype = AutoSparse(AutoForwardDiff())

    cache_B = sparse_jacobian_cache(adtype, sd, myf_u0!, xnext0, u0)
    B = sparse_jacobian(adtype, cache_B, myf_u0!, xnext0, u0)

    # x0 = copy(kite.integrator.u)
    # prep = prepare_jacobian(myf_x0!, xnext0, backend, simple_x0)
    # jacobian!(myf_x0!, xnext0, A, prep, backend, simple_x0)
    # @time jacobian!(myf_x0!, xnext0, A, prep, backend, simple_x0)
    # jacobian!(myf_u0!, xnext0, B, backend, u0)
    # t = @elapsed jacobian!(myf_x0!, xnext0, A, prep, backend, simple_x0)

    f!(xnext0, x0, simple_x0, u0)
    lin_xnext0 = A * simple_x0 + B * u0
    for (i, unk) in enumerate(unknowns(sys)[idxs])
        println(unk, "\t", simple_x0[i], "\t", xnext0[i], "\t", lin_xnext0[i])
    end
    diff_ulin = norm(xnext0 .- simple_x0)
    diff_lin = norm(lin_xnext0 .- simple_x0)
    # @show lin_xnext0 .- simple_x0
    println("solver: ", backend, "\n\tdiff_ulin: ", diff_ulin, "\n\tdiff_lin: ", diff_lin, "\n\ttime ", t)
    return A, B
end

A, B = jac(backend) # works, big diff

# jac(AutoPolyesterForwardDiff()) # ERROR: LoadError: Cannot determine ordering of Dual tags ForwardDiff.Tag{DiffEqBase.OrdinaryDiffEqTag, Dual{Nothing, Float64, 11}} and Nothing
# jac(AutoFiniteDiff()) # very fast
