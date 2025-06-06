struct HC end
Base.:+(m, ::HC) = (m + m')
Base.:-(m, ::HC) = (m - m')
const hc = HC()

hopping(t, f1, f2) = t * f1'f2 + hc
pairing(Δ, f1, f2) = Δ * f2 * f1 + hc
numberop(f) = f'f
coulomb(f1, f2) = f1' * f1 * f2' * f2
function coulomb(f1::BdGFermion, f2::BdGFermion)
    @warn "Returning zero as Coulomb term for BdGFermions. This message is not displayed again." maxlog = 1
    0 * numberop(f1)
end
function su2_rotation((θ, ϕ))
    pf = mod(ϕ, π) == 0 ? real(exp(1im * ϕ)) : exp(1im * ϕ)
    _su2_rotation(θ, pf)
end
_su2_rotation(θ, pf) = ((s, c) = sincos(θ / 2); @SMatrix [c -s*pf'; s*pf c])

function hopping_rotated(t, (c1up, c1dn), (c2up, c2dn), angles1, angles2)
    Ω = su2_rotation(angles1)' * su2_rotation(angles2)
    c1 = @SVector [c1up, c1dn]
    c2 = @SVector [c2up, c2dn]
    t * c1' * Ω * c2 + hc
end

function pairing_rotated(Δ, (c1up, c1dn), (c2up, c2dn), angles1, angles2)
    m = @SMatrix [0 -1; 1 0]
    Ω = transpose(su2_rotation(angles1)) * m * su2_rotation(angles2)
    c1 = @SVector [c1up, c1dn]
    c2 = @SVector [c2up, c2dn]
    (Δ*permutedims(c1)*Ω*c2)[1] + hc
end

_kitaev_2site(f1, f2; t, Δ, V) = hopping(-t, f1, f2) + V * coulomb(f1, f2) + pairing(Δ, f1, f2)
_kitaev_1site(f; μ) = -μ * numberop(f)

function kitaev_hamiltonian(c; μ, t, Δ, V=0)
    N = length(c)
    indices = collect(keys(c))
    h1s = (_kitaev_1site(c[k]; μ=getvalue(μ, j, N)) for (j, k) in enumerate(indices))
    h2s = (_kitaev_2site(c[k], c[indices[mod1(j + 1, N)]]; t=getvalue(t, j, N; size=2), Δ=getvalue(Δ, j, N; size=2), V=getvalue(V, j, N; size=2)) for (j,k) in enumerate(indices))
    sum(h1s) + sum(h2s)
end


function _BD1_2site((c1up, c1dn), (c2up, c2dn); t, Δ1, V, θϕ1, θϕ2)
    ms = hopping_rotated(t, (c1up, c1dn), (c2up, c2dn), θϕ1, θϕ2) +
         pairing_rotated(Δ1, (c1up, c1dn), (c2up, c2dn), θϕ1, θϕ2)
    if iszero(V)
        return ms
    else
        return ms + V * ((numberop(c1up) + numberop(c1dn)) * (numberop(c2up) + numberop(c2dn)))
    end
end
function _BD1_1site((cup, cdn); μ, h, Δ, U)
    (-μ - h) * numberop(cup) + (-μ + h) * numberop(cdn) +
    pairing(Δ, cup, cdn) + U * coulomb(cup, cdn)
end


getvalue(v::Union{<:AbstractVector,<:Tuple}, i, N; size=1) = v[i]
getvalue(x::Number, i, N; size=1) = 1 <= i <= N + 1 - size ? x : zero(x)

cell(j, b) = map(l -> b[l], filter(isequal(j) ∘ first, collect(keys(b))))

function BD1_hamiltonian(c; μ, h, t, Δ, Δ1, U, V, θ, ϕ)
    M = length(c)
    spatial_indices = unique(map(first, collect(keys(c))))
    @assert length(cell(first(spatial_indices), c)) == 2 "Each unit cell should have two fermions for this hamiltonian"
    N = div(M, 2)
    h1s = (_BD1_1site(cell(l, c); μ=getvalue(μ, j, N), h=getvalue(h, j, N), Δ=getvalue(Δ, j, N), U=getvalue(U, j, N)) for (j, l) in enumerate(spatial_indices))
    h2s = (_BD1_2site(cell(l, c), cell(spatial_indices[mod1(j + 1, N)], c); t=getvalue(t, j, N; size=2), Δ1=getvalue(Δ1, j, N; size=2), V=getvalue(V, j, N; size=2), θϕ1=(getvalue(θ, j, N), getvalue(ϕ, j, N)), θϕ2=(getvalue(θ, mod1(j + 1, N), N), getvalue(ϕ, mod1(j + 1, N), N))) for (j, l) in pairs(IndexLinear(), spatial_indices))
    return sum(h1s) + sum(h2s)
end

##
"""
    struct DiagonalizedHamiltonian{Vals,Vecs,H} <: AbstractDiagonalHamiltonian

A struct representing a diagonalized Hamiltonian.

# Fields
- `values`: The eigenvalues of the Hamiltonian.
- `vectors`: The eigenvectors of the Hamiltonian.
- `original`: The original Hamiltonian.
"""
struct DiagonalizedHamiltonian{Vals,Vecs,H} <: AbstractDiagonalHamiltonian
    values::Vals
    vectors::Vecs
    original::H
end
Base.eltype(::DiagonalizedHamiltonian{Vals,Vecs}) where {Vals,Vecs} = promote_type(eltype(Vals), eltype(Vecs))
Base.size(h::DiagonalizedHamiltonian) = size(eigenvectors(h))
Base.:-(h::DiagonalizedHamiltonian) = DiagonalizedHamiltonian(-h.values, -h.vectors, -h.original)
Base.iterate(S::DiagonalizedHamiltonian) = (S.values, Val(:vectors))
Base.iterate(S::DiagonalizedHamiltonian, ::Val{:vectors}) = (S.vectors, Val(:original))
Base.iterate(S::DiagonalizedHamiltonian, ::Val{:original}) = (S.original, Val(:done))
Base.iterate(::DiagonalizedHamiltonian, ::Val{:done}) = nothing
Base.adjoint(H::DiagonalizedHamiltonian) = DiagonalizedHamiltonian(conj(H.values), adjoint(H.vectors), adjoint(H.original))
original_hamiltonian(H::DiagonalizedHamiltonian) = H.original
