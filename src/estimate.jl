using ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t

ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0
@register_symbolic ssqrt(x)

@mtkmodel QuadtankModel begin
    @parameters begin
        k1 = 1.4
        k2 = 1.4
        g = 9.81
        A = 5.1
        a = 0.03
        γ = 0.25
    end
    begin
        A1 = A2 = A3 = A4 = A
        a1 = a3 = a2 = a4 = a
        γ1 = γ2 = γ
    end
    @variables begin
        h(t)[1:4] = 0
        u(t)[1:2] = 0
    end
    @equations begin
        D(h[1]) ~ -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        D(h[2]) ~ -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        D(h[3]) ~ -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        D(h[4]) ~ -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    end
end

@named mtkmodel = QuadtankModel()
mtkmodel = complete(mtkmodel)
@show f, x_sym, ps = ModelingToolkit.generate_control_function(mtkmodel)
