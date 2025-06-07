# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

using ModelPredictiveControl, Plots
using JuMP, DAQP
daqp = Model(DAQP.Optimizer, add_bridges=false)

function pendulum(par, x, u)
    g, L, K, m = par        # [m/s²], [m], [kg/s], [kg]
    θ, ω = x[1], x[2]       # [rad], [rad/s]
    τ  = u[1]               # [Nm]
    dθ = ω
    dω = -g/L*sin(θ) - K/m*ω + τ/m/L^2
    return [dθ, dω]
end
# declared constants, to avoid type-instability in the f function, for speed:
const par = (9.8, 0.4, 1.2, 0.3)
f(x, u, _, _ ) = pendulum(par, x, u)
h(x, _, _ )    = [x[1], 180/π*(x[1]+x[2])]  # [°]
nu, nx, ny, Ts = 1, 2, 2, 0.1
vu, vx, vy = ["\$τ\$ (Nm)"], ["\$θ\$ (rad)", "\$ω\$ (rad/s)"], ["θ", "\$θ\$ (°)"]
model = setname!(NonLinModel(f, h, Ts, nu, nx, ny); u=vu, x=vx, y=vy)

Hp, Hc, Mwt, Nwt = 5, 2, [0.0, 0.5], [2.5]
α=0.01; σQ=[100000, 100000]; σR=[0.00001, 0.00001]; nint_u=[1]; σQint_u=[0.1]
umin, umax = [-1.5], [+1.5]
u = [0.5]
N = 100
linmodel = linearize(model, x=[0, 0], u=[0])
kf = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
mpc3 = LinMPC(kf; Hp, Hc, Mwt, Nwt, Cwt=Inf, optim=daqp)
mpc3 = setconstraint!(mpc3; umin, umax)

function sim_adapt!(mpc, nonlinmodel, N, ry, plant, x_0, x̂_0, y_step=[0, 0])
    U_data, Y_data, Ry_data, X_data, X̂_data = 
        zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N), zeros(plant.nx, N), zeros(plant.ny+plant.nx, N)
    setstate!(plant, x_0)
    initstate!(mpc, [0], plant())
    # setstate!(mpc, x̂_0)
    for i = 1:N
        y = plant() + y_step
        x̂ = preparestate!(mpc, y)
        u = moveinput!(mpc, ry)
        linmodel = linearize(nonlinmodel; u, x=x̂[1:2])
        @show linmodel.yop
        setmodel!(mpc, linmodel)
        U_data[:,i], Y_data[:,i], Ry_data[:,i], X_data[:, i], X̂_data[:, i] = u, y, ry, plant.x0, x̂
        updatestate!(mpc, u, y) # update mpc state estimate
        updatestate!(plant, u)  # update plant simulator
    end
    res = SimResult(mpc, U_data, Y_data; Ry_data, X_data, X̂_data)
    return res
end

p_plant = [9.8, 0.4, 1.2*1.25, 0.3]
plant = setname!(NonLinModel(f, h, Ts, nu, nx, ny; p=p_plant); u=vu, x=vx, y=vy)
x_0 = [0, 0]; x̂_0 = [0, 0, 0]; ry = [0, 180]
res_slin = sim_adapt!(mpc3, model, N, ry, plant, x_0, x̂_0)
plot(res_slin; plotxwithx̂=true)