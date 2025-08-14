using SymbolicAWEModels
using ModelPredictiveControl
using ControlSystems
using Plots
using UnPack
using ModelingToolkit
import ModelingToolkit: t_nounits as t

@variables begin
    heading(t)[1]
    turn_rate(t)[1, 1:3]
    elevation(t)[1]
    azimuth(t)[1]
    elevation_vel(t)[1]
    azimuth_vel(t)[1]
    tether_len(t)[1:3]
    tether_vel(t)[1:3]
    winch_force(t)[1:3]
    angle_of_attack(t)[1]
end
lin_outputs = [
    # measured
    heading[1],
    turn_rate[1,3],
    elevation[1],
    azimuth[1],
    elevation_vel[1],
    azimuth_vel[1],
    tether_len...,
    tether_vel...,

    # not measured
    winch_force...,
    angle_of_attack[1],
]
@info "Linear outputs: $lin_outputs"

set = Settings("system.yaml")
dt = 1/set.sample_freq
sam = SymbolicAWEModel(set, "ram")
init!(sam; lin_outputs)

simple_plant_set = Settings("system.yaml")
simple_plant_sam = SymbolicAWEModel(simple_plant_set, "simple_ram")
init!(simple_plant_sam; lin_outputs)

tether_set = Settings("system.yaml")
tether_sam = SymbolicAWEModel(tether_set, "tether")
init!(tether_sam)

simple_set = Settings("system.yaml")
simple_sam = SymbolicAWEModel(simple_set, "simple_ram")
init!(simple_sam; lin_outputs, create_control_func=true)

copy_to_simple!(sam, tether_sam, simple_sam; prn=false)
copy_to_simple!(sam, tether_sam, simple_plant_sam; prn=false)

function linearize(simple_sam)
    # Set y to sam, copy sam to simple_sam, linearize simple sam
    # TODO: set y to sam
    u = -simple_sam.set.drum_radius .*
        [norm(winch.force) for winch in simple_sam.sys_struct.winches]
    statespace = SymbolicAWEModels.linearize!(simple_sam)
    statespace = ControlSystems.ss(statespace...)
    statespace = c2d(statespace, dt)
    @unpack nx, ny, nu = statespace
    Bd = zeros(nx, 0)
    Dd = zeros(ny, 0)
    A = statespace.A
    Bu = statespace.B
    C = statespace.C
    @assert norm(statespace.D) ≈ 0
    linmodel = setname!(LinModel{Float64}(A, Bu, C, Bd, Dd, dt),
                        x=string.(unknowns(simple_sam.sys)), y=string.(lin_outputs))

    # --- compute the nonlinear model output at operating points ---
    y = simple_sam.get_lin_y(simple_sam.integrator)
    # --- modify the linear model operating points ---
    linmodel.uop .= u
    @show ny length(y)
    linmodel.yop .= y
    linmodel.xop .= simple_sam.integrator.u
    # --- compute the nonlinear model next state at operating points ---
    next_step!(simple_sam; set_values=u)
    linmodel.fop .= simple_sam.integrator.u # TODO: investigate for sudden jump in x
    # --- reset the state of the linear model ---
    linmodel.x0 .= 0 # state deviation vector is always x0=0 after a linearization
    return linmodel
end

@time linmodel = linearize(simple_sam)
nx, ny, nu = linmodel.nx, linmodel.ny, linmodel.nu
# α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = KalmanFilter(linmodel)

Hp, Hc = 20, 2
# Mwt, Nwt = [0.5], [2.5]
mpc = LinMPC(estim; Hp, Hc, Cwt=Inf)
umin = fill(-100, 3)
umax = fill(100, 3)
mpc = setconstraint!(mpc; umin, umax)

# function sim_adapt!(mpc, N, ry, x_0, x̂_0, y_step=zeros(ny))
#     U_data, Y_data, Ry_data = zeros(nu, N), zeros(ny, N), zeros(ny, N)
#     initstate!(mpc, [0], sam.get_lin_y(sam.integrator))
#     setstate!(mpc, x̂_0)
#     for i = 1:N
#         y = sam.get_lin_y(sam.integrator) + y_step
#         x̂ = preparestate!(mpc, y)
#         u = moveinput!(mpc, ry)
#         # linmodel = linearize!(simple_sam, x̂)
#         # setmodel!(mpc, linmodel)
#         U_data[:,i], Y_data[:,i], Ry_data[:,i] = u, y, ry
#         updatestate!(mpc, u, y) # update mpc state estimate
#         next_step!(sam; set_values=u)  # update plant simulator
#     end
#     res = SimResult(mpc, U_data, Y_data; Ry_data)
#     return res
# end
