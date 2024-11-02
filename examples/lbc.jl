"""
NOT WORKING

GeneralLazyBufferCache as was implemented in mtk_interface.jl. Is needed when using forwarddiff.
"""

get_y = getu(kite.integrator, outputs)
    
# --- The outputs are heading, depower and tether length, and they can be calculated from this state ---
sys = kite.prob.f.sys
simple_state = outputs
state = unknowns(sys)
nu = length(inputs)
nx_simple = length(simple_state) + 1
nx = length(state)
ny = length(outputs)

function make_default_creator(state)
    keys = collect(state)
    return x -> Dict(k => x[i] for (i, k) in enumerate(keys))
end
create_default = make_default_creator(state)

integ_cache = GeneralLazyBufferCache(
    function (xu)
        x = xu[1:nx]
        u = xu[nx+1:end]
        default = create_default(x)
        par = vcat([inputs[i] => u[i] for i in 1:nu])
        prob = ODEProblem(sys, default, (0.0, Ts), par)
        setu! = setp(prob, [inputs[i] for i in 1:nu])
        get_simple_x = getu(prob, simple_state)
        integrator = OrdinaryDiffEqCore.init(prob, solver; saveat=Ts, abstol=kite.set.abs_tol, reltol=kite.set.rel_tol, verbose=false)
        return (integrator, setu!, get_simple_x)
    end
)

"""
Nonlinear discrete dynamics. Takes in complex state and returns simple state_plus
"""
function next_step!(x_simple_plus, x, u, dt, integ_setu_pair)
    (integ, setu!, get_simple_x) = integ_setu_pair
    reinit!(integ, x; t0=1.0, tf=1.0+dt)
    setu!(integ, u)
    OrdinaryDiffEqCore.step!(integ, dt, true)
    @assert successful_retcode(integ.sol)
    x_simple_plus[1:nx_simple-1] .= get_simple_x(integ)
    x_simple_plus[end] = 1
    return x_simple_plus
end
function f!(x_simple_plus, x, u, dt)
    next_step!(x_simple_plus, x, u, dt, integ_cache[vcat(x, u)])
end 

"Observer function"
function get_y!(y, x, integ_setu_pair)
    (integ, _, _) = integ_setu_pair
    reinit!(integ, x; t0=1.0, tf=1.0+Ts)
    y .= get_y(integ)
    return y
end
h!(y, x) = get_y!(y, x, integ_cache[vcat(x, zeros(3))])