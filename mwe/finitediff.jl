using ModelPredictiveControl, FiniteDiff, DifferentiationInterface

ad_type = AutoFiniteDiff()

function f(x, u, _ , p)
    g, L, K, m = p          # [m/s²], [m], [kg/s], [kg]
    θ, ω = x[1], x[2]       # [rad], [rad/s]
    τ  = u[1]               # [Nm]
    dθ = ω
    dω = -g/L*sin(θ) - K/m*ω + τ/m/L^2
    return [dθ, dω]
end
h(x, _ , _ ) = [180/π*x[1]] # [°]
p_model = [9.8, 0.4, 1.2, 0.3]
nu, nx, ny, Ts = 1, 2, 1, 0.1
vu, vx, vy = ["\$τ\$ (Nm)"], ["\$θ\$ (rad)", "\$ω\$ (rad/s)"], ["\$θ\$ (°)"]
model = setname!(NonLinModel(f, h, Ts, nu, nx, ny; p=p_model, jacobian=ad_type); u=vu, x=vx, y=vy)


using Plots
u = [0.5]
N = 35
res = sim!(model, N, u)
plot(res, plotu=false)


α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)


p_plant = copy(p_model)
p_plant[3] = 1.25*p_model[3]
plant = setname!(NonLinModel(f, h, Ts, nu, nx, ny; p=p_plant, jacobian=ad_type); u=vu, x=vx, y=vy)
res = sim!(estim, N, [0.5], plant=plant, y_noise=[0.5])
plot(res, plotu=false, plotxwithx̂=true)

# linmodel = ModelPredictiveControl.linearize(model, x=[π, 0], u=[0])

Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]
nmpc = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf, transcription=MultipleShooting(), gradient=ad_type, jacobian=ad_type)
umin, umax = [-1.5], [+1.5]
nmpc = setconstraint!(nmpc; umin, umax)


res_ry = sim!(nmpc, N, [180.0], plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
plot(res_ry)