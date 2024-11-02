"""

torque control with uop = -winch_force(s)*0.11

1.
heading_y'   = turn_rate_y = k * flap_diff

    set_value[2] = 5.0
    set_value[1] = -5.0
    next_step!(kite; dt="at least 0.2, doesnt matter that much")
    k = turn_rate / flap_diff

2.
flap_diff'  = flap_diff_vel = k1 * tether_diff'
            = k1 * set_value[1] - k1 * set_value[2])
    set_value[2] = 5.0
    set_value[1] = -5.0
    next_step!(kite; dt="1/3 of Hp")
    k1 = tether_diff_vel / (set_value[1] - set_value[2])

3.
tether_length' = k * set_value
    set_value .= 5.0
    step
    k = tether_vel / set_value

one' = 0.0

size(X) = (nm, nsx)
size(U) = (nm, nu)
size(X_plus) = (nm, nsx)
X_plus' = [A B] * [X' ; U']
[A B] = [X' ; U'] '\' X_plus'
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
    A_pattern[idx(sys.turn_rate_y), idx(sys.flap_diff)] = 1.0

    # --- flap diff ---
    A_pattern[idx(sys.flap_diff), idx(sys.tether_diff_vel)] = 1.0
    A_pattern[idx(sys.flap_diff), idx(sys.set_diff)] = 1.0
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


function linearize(sys, s_idxs, m_idxs, measure_f!, simple_f!, nsx, nmx, nu, xname, uname, x0, x_simple_0, u0, Ts, Hp)
    A = zeros(nsx, nsx)
    Bu = zeros(nsx, nu)
    C = I(nsx)
    Bd = zeros(nsx, 0)
    Dd = zeros(nsx, 0)

    du = 1.0
    U = [
        -du     du      0.0 # left
        du      -du     0.0 # right
        -du     -du     -du  # in
    ]
    X = zeros(size(U, 1), nmx)
    X_plus = similar(X)
    measurements = (X_plus, X, U)

    linmodel = LinModel{Float64}(A, Bu, C, Bd, Dd, Ts)
    linmodel.uname .= uname
    linmodel.xname .= xname
    linmodel.yname .= linmodel.xname

    return linearize!(linmodel, sys, s_idxs, m_idxs, measure_f!, simple_f!, measurements, x0, x_simple_0, u0, Ts, Hp), measurements
end

"""
Linearize using the known sparsity pattern (no sparse matrix as this allocates more).
"""
function linearize!(ci::ControlInterface, linmodel::LinModel, x0, u0)
    return linearize!(linmodel, ci.f!, ci.h!, ci.AB, ci.AB_pattern, x0, ci.x_simple_0, u0)
end
function linearize!(linmodel::LinModel, sys, s_idxs, m_idxs, measure_f!, simple_f!, measurements, x0, x_simple_0, u0, Ts, Hp)
    (X_plus, X, U) = measurements
    A, B, C, D = linmodel.A, linmodel.Bu, linmodel.C, linmodel.Dd
    A .= 0.0
    B .= 0.0

    @show Ts*Hp/3
    for i in eachindex(U[:, 1])
        x_plus = @view X_plus[i, :]
        measure_f!(x_plus, x0, U[i, :], Ts*Hp/3)
    end

    # --- heading ---
    i = abs(X_plus[1, m_idxs[sys.flap_diff]]) > abs(X_plus[2, m_idxs[sys.flap_diff]]) ?
        1 : 2
    A[s_idxs[sys.heading_y], s_idxs[sys.flap_diff]] =
            X_plus[i, m_idxs[sys.turn_rate_y]] / X_plus[i, m_idxs[sys.flap_diff]]

    # --- flap angle difference ---
    i = (X_plus[1, m_idxs[sys.tether_diff_vel]]) > (X_plus[2, m_idxs[sys.tether_diff_vel]]) ?
        1 : 2
    k = X_plus[i[1], m_idxs[sys.tether_diff_vel]] / (U[i, 1] - U[i, 2])
    B[s_idxs[sys.flap_diff], [1, 2]] .= [k, -k]

    # --- tether velocity ---
    for i in 1:3
        B[s_idxs[sys.tether_length[i]], i] =
            X_plus[3, m_idxs[sys.tether_vel[i]]] / U[3, i]
    end

    @show A, B, C, D

    css = ss(A, B, C, 0)
    dss = c2d(css, Ts, :zoh)
    linmodel.A .= dss.A
    linmodel.Bu .= dss.B

    linmodel.uop .= u0
    linmodel.yop .= x_simple_0 # outputs = states
    linmodel.xop .= x_simple_0
    simple_f!(linmodel.fop, x0, u0, Ts)
    # linmodel.fop .= linmodel.A * x_simple_0 + linmodel.Bu * u0
    linmodel.x0 .= 0
    return linmodel
end


# # Update only the non-zero elements of A and B
# A.nzval .= nonzeros(A_new)
# B.nzval .= nonzeros(B_new)
