# SPDX-FileCopyrightText: 2025 Bart van de Lint
#
# SPDX-License-Identifier: MPL-2.0

using LinearAlgebra
using Printf
using Test # Import the Test package

# Define types for clarity
const Quaternion = Vector{Float64}
const Vec3 = Vector{Float64}

# --- Quaternion Operations ---

"""
    q_mult(q1::Quaternion, q2::Quaternion)::Quaternion

Multiplies two quaternions (q1 * q2) using the Hamilton product.
Assumes q = [w, x, y, z].
"""
function q_mult(q1::Quaternion, q2::Quaternion)::Quaternion
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return [w, x, y, z]
end

"""
    q_conj(q::Quaternion)::Quaternion

Computes the conjugate of a quaternion.
"""
function q_conj(q::Quaternion)::Quaternion
    w, x, y, z = q
    return [w, -x, -y, -z]
end

"""
    q_from_axis_angle(a::Vec3, theta::Float64)::Quaternion

Creates a quaternion representing a rotation of `theta` radians
around the axis `a`.
"""
function q_from_axis_angle(a::Vec3, theta::Float64)::Quaternion
    a_norm = normalize(a)
    ax, ay, az = a_norm
    half_theta = theta / 2.0
    s = sin(half_theta)
    c = cos(half_theta)
    return [c, ax * s, ay * s, az * s]
end

"""
    q_to_z_axis(q::Quaternion)::Vec3

Computes the body's z-axis ([0, 0, 1]) in the global frame
given its orientation quaternion `q`.
"""
function q_to_z_axis(q::Quaternion)::Vec3
    w, x, y, z = normalize(q) # Ensure unit for calculation
    zb_x = 2 * (x * z + w * y)
    zb_y = 2 * (y * z - w * x)
    zb_z = w^2 - x^2 - y^2 + z^2
    return [zb_x, zb_y, zb_z]
end

"""
    q_rot(q::Quaternion, v::Vec3)::Vec3

Rotates a vector `v` by quaternion `q` using q * v * q_conj.
"""
function q_rot(q::Quaternion, v::Vec3)::Vec3
    vx, vy, vz = v
    v_q = [0.0, vx, vy, vz]
    q_norm = normalize(q)
    q_inv = q_conj(q_norm)
    v_rot_q = q_mult(q_norm, q_mult(v_q, q_inv))
    return [v_rot_q[2], v_rot_q[3], v_rot_q[4]]
end


# --- Core Logic ---

"""
    adjust_quaternion(Q::Quaternion, z2::Vec3, a::Vec3; tol::Float64 = 1e-8, verbose::Bool = false)

Adjusts the quaternion `Q` such that its body z-axis aligns with `z2`
by rotating only around axis `a`.
"""
function adjust_quaternion(Q::Quaternion, z2::Vec3, a::Vec3; tol::Float64 = 1e-8, verbose::Bool = false)::Union{Quaternion, Nothing}
    a_norm = normalize(a)
    z2_norm = normalize(z2)
    Q_norm = normalize(Q)

    zb = q_to_z_axis(Q_norm)
    zb = normalize(zb)

    dot_b = dot(zb, a_norm)
    dot_2 = dot(z2_norm, a_norm)

    if !isapprox(dot_b, dot_2, atol=tol)
        if verbose @printf("Alignment impossible: Vectors have different angles with the rotation axis (%.4f vs %.4f).\n", acos(clamp(dot_b,-1,1)), acos(clamp(dot_2,-1,1))) end
        return nothing
    end

    zb_perp = zb - dot_b * a_norm
    z2_perp = z2_norm - dot_2 * a_norm

    norm_b_perp = norm(zb_perp)
    norm_2_perp = norm(z2_perp)

    theta = 0.0

    if norm_b_perp < tol || norm_2_perp < tol
        dot_zb_z2 = dot(zb, z2_norm)
        if dot_zb_z2 > 1.0 - tol
            if verbose @printf("Vectors already aligned (or parallel to 'a' and aligned).\n") end
            theta = 0.0
        elseif dot_zb_z2 < -1.0 + tol
            if verbose @printf("Vectors anti-aligned and parallel to 'a'. Needs 180 deg rotation.\n") end
            theta = pi
        else
            if verbose println("Alignment impossible: Parallel to 'a' but not aligned/anti-aligned.") end
            return nothing
        end
    else
        ub = normalize(zb_perp)
        u2 = normalize(z2_perp)
        cos_theta = clamp(dot(ub, u2), -1.0, 1.0)
        cross_val = cross(ub, u2)
        sin_theta = dot(cross_val, a_norm)
        theta = atan(sin_theta, cos_theta)
        if verbose @printf("Rotation angle needed: %.2f degrees.\n", rad2deg(theta)) end
    end

    Q_rot = q_from_axis_angle(a_norm, theta)
    Q_new = q_mult(Q_rot, Q_norm)
    return normalize(Q_new)
end

# --- Testing with Test.jl ---

"""
    run_tests_with_test_jl()

Runs a series of tests using the `Test.jl` package.
"""
function run_tests_with_test_jl()
    tol = 1e-6

    @testset "Quaternion Adjustment Tests" begin

        @testset "Test 1: Prompt Example" begin
            Q1 = [1.0, 0.0, 0.0, 0.0]
            z2_1 = [1.0, 0.0, 0.0]
            a1 = [0.0, 1.0, 0.0]
            Q_new1 = adjust_quaternion(Q1, z2_1, a1)
            @test Q_new1 !== nothing
            zb_new1 = q_to_z_axis(Q_new1)
            @test zb_new1 ≈ z2_1 atol = tol
            @test Q_new1 ≈ [sqrt(2)/2, 0, sqrt(2)/2, 0] atol = tol
        end

        @testset "Test 2: Already Aligned" begin
            Q2 = [0.70710678, 0.70710678, 0.0, 0.0] # ~90 deg around X
            zb2 = q_to_z_axis(Q2) # Should be ~[0, 1, 0]
            z2_2 = zb2
            a2 = [0.0, 0.0, 1.0]
            Q_new2 = adjust_quaternion(Q2, z2_2, a2)
            @test Q_new2 !== nothing
            zb_new2 = q_to_z_axis(Q_new2)
            @test zb_new2 ≈ z2_2 atol = tol
            # Since theta should be 0, Q_new should be Q
            @test Q_new2 ≈ normalize(Q2) atol = tol
        end

        @testset "Test 3: 180 Degrees" begin
            Q3 = [1.0, 0.0, 0.0, 0.0] # z_b = [0, 0, 1]
            z2_3 = [0.0, 0.0, -1.0]
            a3 = [0.0, 1.0, 0.0] # Rotate around Y
            Q_new3 = adjust_quaternion(Q3, z2_3, a3)
            @test Q_new3 !== nothing
            zb_new3 = q_to_z_axis(Q_new3)
            @test zb_new3 ≈ z2_3 atol = tol
            # 180 deg around Y is [0, 0, 1, 0]
            @test Q_new3 ≈ [0.0, 0.0, 1.0, 0.0] atol = tol
        end

        @testset "Test 4: Impossible Case (Angle Mismatch)" begin
            Q4 = [1.0, 0.0, 0.0, 0.0] # z_b = [0, 0, 1]
            z2_4 = [1.0, 0.0, 0.0]
            a4 = [0.0, 0.0, 1.0] # Rotate around Z
            Q_new4 = adjust_quaternion(Q4, z2_4, a4)
            @test isnothing(Q_new4)
        end

        @testset "Test 5: General Case" begin
            Q5 = q_from_axis_angle([1.0, 1.0, 0.0], pi/4) # Initial Q
            zb5 = q_to_z_axis(Q5)
            a5 = [0.0, 0.0, 1.0] # Rotate around Z
            Q_rot_test = q_from_axis_angle(a5, pi/3) # 60 deg rot
            z2_5 = q_rot(Q_rot_test, zb5)
            Q_new5 = adjust_quaternion(Q5, z2_5, a5)
            @test Q_new5 !== nothing
            zb_new5 = q_to_z_axis(Q_new5)
            @test zb_new5 ≈ z2_5 atol = tol
            # Q_new should be Q_rot_test * Q5
            @test Q_new5 ≈ q_mult(Q_rot_test, Q5) atol = tol
        end

        @testset "Test 6: Impossible Case (Parallel & Mismatch)" begin
            Q6 = [0.0, 1.0, 0.0, 0.0] # 180 deg around X, zb = [0, 0, -1]
            a6 = [0.0, 0.0, 1.0] # Rotate around Z
            z2_6 = [0.0, 0.0, 1.0]
            # zb.a = -1, z2.a = 1 -> Impossible
            Q_new6 = adjust_quaternion(Q6, z2_6, a6)
            @test isnothing(Q_new6)
        end

        @testset "Test 7: zb parallel to 'a' (0 deg)" begin
            Q7 = [1.0, 0.0, 0.0, 0.0] # zb = [0, 0, 1]
            a7 = [0.0, 0.0, 1.0] # Rotate around Z
            z2_7 = [0.0, 0.0, 1.0]
            # zb.a = 1, z2.a = 1. Projections are 0. zb = z2. Theta = 0.
            Q_new7 = adjust_quaternion(Q7, z2_7, a7)
            @test Q_new7 !== nothing
            zb_new7 = q_to_z_axis(Q_new7)
            @test zb_new7 ≈ z2_7 atol = tol
            @test Q_new7 ≈ Q7 atol = tol
        end

    end # End of main testset
end

# Example of running the function with verbose output
println("--- Running Example with Verbose Output ---")
Q_ex = [1.0, 0.0, 0.0, 0.0]
z2_ex = [1.0, 0.0, 0.0]
a_ex = [0.0, 1.0, 0.0]
Q_new_ex = adjust_quaternion(Q_ex, z2_ex, a_ex, verbose=true)
if Q_new_ex !== nothing
    zb_new_ex = q_to_z_axis(Q_new_ex)
    @printf("Initial Q: [%.3f, %.3f, %.3f, %.3f]\n", Q_ex...)
    @printf("Target z2: [%.3f, %.3f, %.3f]\n", z2_ex...)
    @printf("Axis a:    [%.3f, %.3f, %.3f]\n", a_ex...)
    @printf("New Q:     [%.3f, %.3f, %.3f, %.3f]\n", Q_new_ex...)
    @printf("New zb:    [%.3f, %.3f, %.3f]\n", zb_new_ex...)
end
println("-----------------------------------------\n")


# Run the tests using Test.jl
println("--- Running Tests with Test.jl ---")
run_tests_with_test_jl()
println("----------------------------------")
