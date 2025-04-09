using ModelPredictiveControl, DifferentiationInterface, FiniteDiff
using Plots

ad_type = AutoFiniteDiff(relstep=0.1, absstep=0.1)

function f!(dx, x, u, _ , p)
    g, L, K, m = p          # [m/s²], [m], [kg/s], [kg]
    θ, ω = x[1], x[2]       # [rad], [rad/s]
    τ  = u[1]               # [Nm]
    dθ = ω
    dω = -g/L*sin(θ) - K/m*ω + τ/m/L^2
    dx[1] = dθ
    dx[2] = dω
    return nothing
end
function h!(y, x, _ , _ ) 
    y .= [180/π*x[1]] # [°]
    nothing
end
p_model = [9.8, 0.4, 1.2, 0.3]
nu, nx, ny, Ts = 1, 2, 1, 0.1
vu, vx, vy = ["\$τ\$ (Nm)"], ["\$θ\$ (rad)", "\$ω\$ (rad/s)"], ["\$θ\$ (°)"]
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p=p_model, jacobian=ad_type); u=vu, x=vx, y=vy)

u = [0.5]
N = 35

α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)

p_plant = copy(p_model)
p_plant[3] = 1.25*p_model[3]
plant = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p=p_plant, jacobian=ad_type); u=vu, x=vx, y=vy)
res = sim!(estim, N, [0.5], plant=plant, y_noise=[0.5])
plot(res, plotu=false, plotxwithx̂=true)

# Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]
# umin, umax = [-1.5], [+1.5]

# linmodel = ModelPredictiveControl.linearize(model, x=[0, 0], u=[0])
# display(linmodel.A); display(linmodel.Bu)
# kf = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
# mpc3 = LinMPC(kf; Hp, Hc, Mwt, Nwt, Cwt=Inf)
# mpc3 = setconstraint!(mpc3; umin, umax)

# function sim_adapt!(mpc, nonlinmodel, N, ry, plant, x_0, x̂_0, y_step=[0])
#     U_data, Y_data, Ry_data = zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N)
#     setstate!(plant, x_0)
#     initstate!(mpc, [0], plant())
#     setstate!(mpc, x̂_0)
#     for i = 1:N
#         y = plant() + y_step
#         x̂ = preparestate!(mpc, y)
#         u = moveinput!(mpc, ry)
#         linmodel = ModelPredictiveControl.linearize(nonlinmodel; u, x=x̂[1:2])
#         setmodel!(mpc, linmodel)
#         U_data[:,i], Y_data[:,i], Ry_data[:,i] = u, y, ry
#         updatestate!(mpc, u, y) # update mpc state estimate
#         updatestate!(plant, u)  # update plant simulator
#     end
#     res = SimResult(mpc, U_data, Y_data; Ry_data)
#     return res
# end

# x_0 = [0, 0]; x̂_0 = [0, 0, 0]; ry = [180]
# res_slin = sim_adapt!(mpc3, model, N, ry, plant, x_0, x̂_0)
# plot(res_slin)
