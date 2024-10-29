"""
heading_y'   = turn_rate_y    +   0 * set_value
turn_rate_y' = k_1 * one      +   k_4 * set_value

flap_angle' = flap_vel          +   0 * set_value
flap_vel'   = acc = k_2 * one   +   k_5 * set_value

tether_length'  = tether_vel        +   0 * set_value
tether_vel'     = acc = k_3 * one   +   k_6 * set_value

one' = 0.0


X_prime = A * X + B * U

... * A = X_prime


c2d
"""

# time_multiplier = 10
# (f!, h!, simple_state, nu, _, nsx, ny) = generate_f_h(kite, inputs, outputs, solver, Ts * time_multiplier)


function generate_AB(sys, nsx, nu, xname)
    # function idx(symbol)
    #     return idx(xname, symbol)
    # end
    idx = var -> findfirst(x -> x == string(var), xname)

    A_pattern = zeros(nsx, nsx)
    B_pattern = zeros(nsx, nu)

    # --- heading ---
    A_pattern[idx(sys.heading_y), idx(sys.turn_rate_y)] = 1.0
    A_pattern[idx(sys.heading_y), nsx] = 1.0
    B_pattern[idx(sys.heading_y), [1, 2]] .= 1.0
    A_pattern[idx(sys.turn_rate_y), nsx] = 1.0
    B_pattern[idx(sys.turn_rate_y), [1, 2]] .= 1.0

    # --- depower ---
    A_pattern[idx(sys.depower), idx(sys.depower_vel)] = 1.0
    A_pattern[idx(sys.depower), nsx] = 1.0
    B_pattern[idx(sys.depower), 1:3] .= 1.0
    A_pattern[idx(sys.depower_vel), nsx] = 1.0
    B_pattern[idx(sys.depower_vel), 1:3] .= 1.0

    # --- tether length ---
    for i in 1:3
        A_pattern[idx(sys.tether_length[i]), idx(sys.tether_vel[i])] = 1.0
        A_pattern[idx(sys.tether_length[i]), nsx] = 1.0
        B_pattern[idx(sys.tether_length[i]), i] = 1.0
        A_pattern[idx(sys.tether_vel[i]), nsx] = 1.0
        B_pattern[idx(sys.tether_vel[i]), i] = 1.0
    end

    AB_pattern = [A_pattern B_pattern]
    AB = copy(AB_pattern)
    return AB, AB_pattern
end


function linearize(sys, f!, h!, nsx, nu, xname, uname, x0, x_simple_0, u0, Ts; time_multiplier)
    A = zeros(nsx, nsx)
    Bu = zeros(nsx, nu)
    C = I(nsx)
    Bd = zeros(nsx, 0)
    Dd = zeros(nsx, 0)

    linmodel = LinModel{Float64}(A, Bu, C, Bd, Dd, Ts)
    linmodel.uname .= uname
    linmodel.xname[1:end-1] .= xname
    linmodel.xname[end] = "one"
    linmodel.yname .= linmodel.xname

    AB, AB_pattern = generate_AB(sys, nsx, nu, xname)
    return linearize!(linmodel, f!, h!, AB, AB_pattern, x0, x_simple_0, u0, Ts; time_multiplier), AB, AB_pattern
end

"""
Linearize using the known sparsity pattern (no sparse matrix as this allocates more).
"""
function linearize!(ci::ControlInterface, linmodel::LinModel, x0, u0)
    return linearize!(linmodel, ci.f!, ci.h!, ci.AB, ci.AB_pattern, x0, ci.x_simple_0, u0, ci.Ts; ci.time_multiplier)
end
function linearize!(linmodel::LinModel, f!, h!, AB::Matrix, AB_pattern::Matrix, x0, x_simple_0, u0, Ts; time_multiplier=10.0)
    nu = linmodel.nu
    nsx = linmodel.nx
    x_simple_plus = linmodel.buffer.x
    h!(x_simple_0, x0)
    steering_u = [-5.0, 5.0]
    middle_u = [-20.0, 20.0]
    nm = length(middle_u) * nu # number of measurements
    U = zeros(nm, nu)
    X = zeros(nm, nsx)
    X_prime = zeros(nm, nsx)
    for i in 1:nu
        for j in eachindex(middle_u)
            U[i, :] .= u0 + [i == 1 ? steering_u[j] : 0.0, i == 2 ? steering_u[j] : 0.0, i == 3 ? middle_u[j] : 0.0]
            f!(x_simple_plus, x0, U[i, :], Ts * time_multiplier)
            X[i, :] .= x_simple_0
            X_prime[i, 1:nsx] .= (x_simple_plus .- x_simple_0) / time_multiplier
        end
    end
    U[end, :] .= u0
    f!(x_simple_plus, x0, U[end, :], Ts)
    X[end, :] .= x_simple_0
    X_prime[end, 1:nsx] .= x_simple_plus .- x_simple_0
    XU = [X U]

    # --- solve ---
    for i in 1:nsx
        if !isempty(findall(!iszero, AB_pattern[i, nsx+1:end]))
            optimize_idxs = findall(!iszero, AB_pattern[i, :])
            Z = XU[:, optimize_idxs]
            "AB * XU = X_prime"
            AB[i, optimize_idxs] .= Z \ X_prime[:, i]
        end
    end
    # css = ss(AB[:, 1:nsx], AB[:, nsx+1:end], C, D)
    # dss = c2d(css, Ts)
    linmodel.A .= AB[:, 1:nsx] .+ I(nsx)
    linmodel.Bu .= AB[:, nsx+1:end]

    linmodel.uop .= u0
    linmodel.yop .= h!(linmodel.buffer.y, x0)
    linmodel.xop .= x_simple_0
    linmodel.fop .= x_simple_plus
    # linmodel.fop .= linmodel.A * x_simple_0 + linmodel.Bu * u0
    linmodel.x0 .= 0
    return linmodel
end


# # Update only the non-zero elements of A and B
# A.nzval .= nonzeros(A_new)
# B.nzval .= nonzeros(B_new)
