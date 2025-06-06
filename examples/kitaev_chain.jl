using QuantumDots, LinearAlgebra, SparseArrays
# using Symbolics

@fermions f
kitaev_chain(N, f, μ, t, Δ, U) = sum(t * f[i]' * f[i+1] + hc for i in 1:N-1) +
                                 sum(Δ * f[i] * f[i+1] + hc for i in 1:N) +
                                 sum(U * f[i]' * f[i] * f[i+1]' * f[i+1] for i in 1:N-1) +
                                 sum(μ * f[i]' * f[i] for i in 1:N)
N = 3
H = hilbert_space(1:N)
params = (μ, t, Δ, U) = (1.0, 1.0, 0.5, 2.0)
ham = kitaev_chain(N, f, μ, t, Δ, U)
@time kitaev_chain(100, f, μ, t, Δ, U);
@profview kitaev_chain(100, f, μ, t, Δ, U);
sparse(ham, H)

Hparity = hilbert_space(1:N, ParityConservation())
sparse(ham, Hparity)
