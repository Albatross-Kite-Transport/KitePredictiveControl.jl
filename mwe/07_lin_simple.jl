using SymbolicAWEModels
using ModelPredictiveControl
using ControlSystems
using UnPack

set = Settings("system.yaml")
dt = 1/set.sample_freq
sam = SymbolicAWEModel(set, "ram") # TODO: add plant
init!(sam)

tether_set = Settings("system.yaml")
tether_sam = SymbolicAWEModel(tether_set, "tether")
init!(tether_sam)

simple_set = Settings("system.yaml")
simple_sam = SymbolicAWEModel(simple_set, "simple_ram")
init!(simple_sam)

function linearize(simple_sam, sam, tether_sam)
    # Set y to sam, copy sam to simple_sam, linearize simple sam
    # TODO: set y to sam
    find_steady_state!(sam)
    uop = -simple_sam.set.drum_radius .*
        [norm(winch.force) for winch in simple_sam.sys_struct.winches]
    copy_to_simple!(sam, tether_sam, simple_sam)
    statespace = SymbolicAWEModels.linearize!(simple_sam)
    statespace = ControlSystems.ss(statespace...)
    statespace = c2d(statespace, dt)
    @unpack nx, ny, nu = statespace
    Bd = zeros(nx, 0)
    Dd = zeros(ny, 0)
    A = statespace.A
    Bu = statespace.B
    C = statespace.C
    linmodel = LinModel{Float64}(A, Bu, C, Bd, Dd, dt)

    # --- compute the nonlinear model output at operating points ---
    y = simple_sam.get_lin_y(simple_sam.integrator)
    # --- modify the linear model operating points ---
    linmodel.uop .= uop
    linmodel.yop .= y
    linmodel.xop .= simple_sam.integrator.u
    # --- compute the nonlinear model next state at operating points ---
    next_step!(simple_sam; set_values=uop)
    linmodel.fop .= simple_sam.integrator.u
    # --- reset the state of the linear model ---
    linmodel.x0 .= 0 # state deviation vector is always x0=0 after a linearization
    return linmodel
end

@time linmodel = linearize(simple_sam, sam, tether_sam)
nx, ny, nu = linmodel.nx, linmodel.ny, linmodel.nu
# α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
# estim = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
# mpc = LinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
# mpc = setconstraint!(mpc; umin, umax)

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
