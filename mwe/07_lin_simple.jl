using SymbolicAWEModels
using ModelPredictiveControl
using ControlSystems
using Plots
using UnPack
using ModelingToolkit
import ModelingToolkit: t_nounits as t

@variables begin
    wing_pos(t)[1, 1:3]
    wing_vel(t)[1, 1:3]
    ω_b(t)[1, 1:3],
    Q_b_w(t)[1, 1:3],
    tether_vel(t)[1:3],
    tether_len(t)[1:3],
    twist_ω(t)[1:2],
    free_twist_angle(t)[1:2]

    heading(t)[1]
    elevation(t)[1]
    azimuth(t)[1]
    winch_force(t)[1:3]
    angle_of_attack(t)[1]
end
# outputs = [
#     # stuff we want to control
#     heading[1],
#     winch_force[1],
#     angle_of_attack[1],

#     # copy of state for observability
#     collect(wing_pos[1,:])...,
#     collect(wing_vel[1,:])...,
#     collect(Q_b_w[1,:])...,
#     collect(ω_b[1,:])...,
#     tether_len...,
#     tether_vel...,
#     free_twist_angle...,
#     twist_ω...,
# ]
outputs = [
    heading[1],
    elevation[1],
    azimuth[1],
    tether_len...,
    tether_vel...,
]
@info "Outputs: $outputs"

set = Settings("system.yaml")
dt = 1/set.sample_freq
sam = SymbolicAWEModel(set, "ram")
init!(sam; outputs)

plant_set = Settings("system.yaml")
plant_sam = SymbolicAWEModel(plant_set, "ram")
init!(plant_sam; outputs)

tether_set = Settings("system.yaml")
tether_sam = SymbolicAWEModel(tether_set, "tether")
init!(tether_sam)

simple_set = Settings("system.yaml")
simple_sam = SymbolicAWEModel(simple_set, "simple_ram")
init!(simple_sam; outputs, create_control_func=true)

copy_to_simple!(sam, tether_sam, simple_sam; prn=false)

function generate_f_h(simple_sam)
    @unpack io_sys, dvs, nx, nu, ny, f_ip, h_ip = simple_sam.control_funcs
    p = ModelingToolkit.get_p(io_sys, [])
    p[3][1] = simple_sam.sys_struct
    p[4][1] = simple_sam.set
    if any(ModelingToolkit.is_alg_equation, equations(io_sys))
        error("Systems with algebraic equations are not supported")
    end
    vx = string.(dvs)
    function f!(ẋ, x, u, _ , p)
        try
            f_ip(ẋ, x, u, p, nothing)
        catch err
            if err isa MethodError
                error("NonLinModel does not support a time argument t in the f function, "*
                      "see the constructor docstring for a workaround.")
            else
                rethrow()
            end
        end
        return nothing
    end
    u_nothing = fill(nothing, nu)
    function h!(y, x, _ , p)
        try
            # MTK.jl supports a `u` argument in `h_ip` function but not this package. We set
            # `u` as a vector of nothing and `h_ip` function will presumably throw an
            # MethodError it this argument is used inside the function
            h_ip(y, x, u_nothing, p, nothing)
        catch err
            if err isa MethodError
                error("NonLinModel only support strictly proper systems (no manipulated "*
                      "input argument u in the output function h)")
            else
                rethrow()
            end
        end
        return nothing
    end
    return f!, h!, p, nu, nx, ny, vx
end
f!, h!, p, nu, nx, ny, vx = generate_f_h(simple_sam)
vu = ["τ[1]", "τ[2]", "τ[3]"]
vy = string.(outputs)
Ts = 1/set.sample_freq
solver = RungeKutta(4; supersample=10)
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p, solver); u=vu, x=vx, y=vy)
man_estim = ManualEstimator(model; nint_u=0, nint_ym=0)
res = ModelPredictiveControl.sim!(model, 100)

struct SAMEstim{GetterType}
    sam::SymbolicAWEModel
    simple_sam::SymbolicAWEModel
    tether_sam::SymbolicAWEModel
    get_x::GetterType
    function SAMEstim(sam, simple_sam, tether_sam)
        get_x = ModelingToolkit.getu(simple_sam.integrator, simple_sam.control_funcs.dvs)
        new{typeof(get_x)}(sam, simple_sam, tether_sam, get_x)
    end
end

function preparestate!(estim::SAMEstim, y::Vector{<:Real})
    @unpack sam, simple_sam, tether_sam, get_x = estim

    # update sam with y
    sam.sys_struct.transforms[1].heading = y[1]
    sam.sys_struct.transforms[1].elevation = y[2]
    sam.sys_struct.transforms[1].azimuth = y[3]
    SymbolicAWEModels.reposition!(sam.sys_struct.transforms, sam.sys_struct)
    i = 4
    for winch in sam.sys_struct.winches
        winch.tether_len = y[i]
        i += 1
    end
    for winch in sam.sys_struct.winches
        winch.tether_vel = y[i]
        i += 1
    end
    @assert i-1 == length(y)

    # copy sam to simple sam
    copy_to_simple!(sam, tether_sam, simple_sam; prn=false)

    return get_x(simple_sam.integrator)
end

function updatestate!(estim::SAMEstim, u::Vector{<:Real}, y::Vector{<:Real})
    sam = estim.sam
    next_step!(sam; set_values=u)
    nothing
end

function updatestate!(sam::SymbolicAWEModel, u::Vector{<:Real})
    next_step!(sam; set_values=u)
    nothing
end

estim = SAMEstim(sam, simple_sam, tether_sam)

Hp, Hc = 10, 2
Mwt = zeros(ny)
Mwt[1] = 1.0
Nwt = fill(0.1, nu)
mpc = NonLinMPC(man_estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
umin = fill(-100, 3)
umax = fill(100, 3)
mpc = setconstraint!(mpc; umin, umax)

function man_sim!(mpc, N, ry, y0, u0, x0)
    U_data, Y_data, Ry_data = zeros(nu, N), zeros(ny, N), zeros(ny, N)
    # initstate!(estim, u0, y0)
    setstate!(mpc, x0)
    for i = 1:N
        @show i
        y = plant_sam.simple_lin_model.get_y(sam.integrator)
        # during preparestate:
        # use reposition! on complex model, update tether len/vel and tune the simple model
        @time x̂ = preparestate!(estim, y)
        ŷ = y # TODO: add a kalman filter to remove noise from y and improve ŷ.
              # this kalman filter should be added inside the custom estim, before
              # running reposition!
        @time setstate!(mpc, x̂)
        @time u = moveinput!(mpc, ry)
        U_data[:,i], Y_data[:,i], Ry_data[:,i] = u, y, ry
        # during updatestate, step with
        ModelPredictiveControl.updatestate!
        @time updatestate!(estim, u, y) # in the estim: step the complex model
        updatestate!(plant_sam, u)  # update plant simulator
    end
    res = SimResult(mpc, U_data, Y_data; Ry_data)
    return res
end


y0 = sam.simple_lin_model.get_y(sam.integrator)
x0 = estim.get_x(estim.simple_sam.integrator)
@show y0
ry = copy(y0)
ry[1] += deg2rad(10)
u0 = sam.prob.get_set_values(sam.integrator)
res = man_sim!(mpc, 100, ry, y0, u0, x0)
plot(res; ploty=[1,2,3])
