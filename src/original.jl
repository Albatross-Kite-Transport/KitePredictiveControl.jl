using ModelPredictiveControl
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
h(x, _, _ )    = [180/π*x[1]]  # [°]
nu, nx, ny, Ts = 1, 2, 1, 0.1
vu, vx, vy = ["\$τ\$ (Nm)"], ["\$θ\$ (rad)", "\$ω\$ (rad/s)"], ["\$θ\$ (°)"]
model = setname!(NonLinModel(f, h, Ts, nu, nx, ny); u=vu, x=vx, y=vy)

using Plots
u = [0.5]
N = 35
res = sim!(model, N, u)
plot(res, plotu=false)

α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)

const par_plant = (par[1], par[2], 1.25*par[3], par[4])
f_plant(x, u, _, _ ) = pendulum(par_plant, x, u)
plant = setname!(NonLinModel(f_plant, h, Ts, nu, nx, ny); u=vu, x=vx, y=vy)
@time res = sim!(estim, N, [0.5], plant=plant, y_noise=[0.5])
plot(res, plotu=false, plotxwithx̂=true)

Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]
nmpc = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
umin, umax = [-1.5], [+1.5]
nmpc = setconstraint!(nmpc; umin, umax)

res_ry = sim!(nmpc, N, [180.0], plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
display(plot(res_ry))

res_yd = sim!(nmpc, N, [180.0], plant=plant, x_0=[π, 0], x̂_0=[π, 0, 0], y_step=[10])
display(plot(res_yd))
