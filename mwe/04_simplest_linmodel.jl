"""
Simplify the nonlinear model to a very simple linear model
States:
    - tether length vel acc
    - heading, turn rate, turn acc
Inputs
    - set torque
Outputs
    - heading
    - angle of attack (linear combination of tether lengths)

Sparsity pattern:
    - Torques to tether acc
    - Tether length to turn acc
"""

using Plots
using KiteModels, KiteUtils, LinearAlgebra
using ModelPredictiveControl, ModelingToolkit, OrdinaryDiffEq, DifferentiationInterface,
    SymbolicIndexingInterface
using ModelingToolkit: setu, setp, getu, getp, @variables
using ModelingToolkit: t_nounits as t
using KiteModels: rotation_matrix_to_quaternion
using SciMLBase: successful_retcode
using RobustAndOptimalControl
using ControlSystems
using Printf

include("utils.jl")
include("plotting.jl")
set_data_path(joinpath(dirname(@__DIR__), "data"))

dt = 0.1
time = 10.0
vsm_dt = 0.1
plot_dt = 0.1
trunc = false

N = Int(round(time÷dt))
vsm_interval = Int(round(vsm_dt÷dt))
plot_interval = Int(round(plot_dt÷dt))

# Initialize model
set_model = deepcopy(load_settings("system_model.yaml"))
set_plant = deepcopy(load_settings("system_plant.yaml"))
set_model.quasi_static = false
set_plant.quasi_static = false
s_model = RamAirKite(set_model)
# s_plant = RamAirKite(set_plant)

measure = Measurement()
measure.sphere_pos .= deg2rad.([70.0 70.0; 1.0 -1.0])
KiteModels.init_sim!(s_model, measure; remake=false, adaptive=true)
sys = s_model.sys
integ = s_model.integrator

winches = [s_model.point_system.winches[i].model for i in 1:3]
d_acc_d_τ = [winches[i].set.drum_radius/winches[i].set.gear_ratio / winches[i].set.inertia_total for i in 1:3]
set.brake == true && error("Winch cannot have brake enabled.")
set.winch_model != "TorqueControlledMachine" && error("Only torque controlled machines are allowed.")

x_vec = [
    sys.heading
    sys.turn_rate
    sys.tether_length[1]
    sys.tether_length[2]
    sys.tether_length[3]
    sys.tether_vel[1]
    sys.tether_vel[2]
    sys.tether_vel[3]
]
y_vec = [
    sys.heading
    sys.tether_length[1]
    sys.angle_of_attack
    sys.winch_force[1]
]
# KiteModels.init_sim!(s_model, measure; remake=false, adaptive=true)

c = collect
get_state = getu(integ, [c(sys.winch_force), c(sys.tether_vel)])

"""
Calculate τ such that tether_acc = 0
"""
function calc_steady_τ(wm::TorqueControlledMachine, winch_force, tether_vel)
    ω = wm.set.gear_ratio/wm.set.drum_radius * tether_vel
    τ_friction = WinchModels.calc_coulomb_friction(wm) * WinchModels.smooth_sign(ω) + WinchModels.calc_viscous_friction(wm, ω)
    steady_τ = -wm.set.drum_radius / wm.set.gear_ratio * winch_force  + τ_friction
    return steady_τ
end

function linearize!(s::RamAirKite, x0, A, B, C, D)
    integ = s.integrator
    winch_force, tether_vel = get_state(integ)
    A .= 0.0
    B .= 0.0
    C .= 0.0
    D .= 0.0

    # D(heading) = turn_rate
    A[1,2] = 1.0
    # D(turn_rate) = d_turn_acc/d_steering * d_steering
    # steering = tether_length[2] - tether_length[3]
    # 
    @assert x0[4] == x0[5]
    function calc_turn_acc(left_length, right_length)
        
    end
    A[2,3] = 
    
    # D(length) = vel
    [A[2+i,5+i] = 1.0 for i in 1:3]
    # D(vel) = d_acc/d_τ * d_τ
    for i in 1:3
        B[5+i,i] = d_acc_d_τ[i]
    end
end

nx = length(x_vec)
A = zeros(nx, nx)
B = zeros(nx, 3)
C = zeros(nx, nx)
D = zeros(nx, 0)
linearize!(s_model, x0, A, B, C, D)

