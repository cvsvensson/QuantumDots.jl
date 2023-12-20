struct QuasiParticle{T,M,L} <: AbstractBdGFermion
    weights::Dictionary{Tuple{L,Symbol},T}
    basis::FermionBdGBasis{M,L}
    function QuasiParticle(weights::Dictionary{Tuple{L,Symbol},T}, basis::FermionBdGBasis{M,L}) where {T,M,L}
        new{T,M,L}(weights, basis)
    end
end
function QuasiParticle(v::AbstractVector{T}, basis::FermionBdGBasis{M,L}) where {T,M,L}
    holelabels = map(k -> (k, :h), keys(basis.position).values)
    particlelabels = map(k -> (k, :p), keys(basis.position).values)
    weights = Dictionary(vcat(holelabels, particlelabels), collect(v))
    QuasiParticle(weights, basis)
end
Base.eltype(qp::QuasiParticle) = eltype(qp.weights)
function Base.getindex(qp::QuasiParticle, i)
    if i in qp.weights.indices
        return getindex(qp.weights, i)
    else
        return zero(eltype(qp))
    end
end
Base.getindex(qp::QuasiParticle, i...) = getindex(qp, i)

function QuasiParticle(f::BdGFermion)
    label = (f.id, f.hole ? :h : :p)
    weights = Dictionary([label], [f.amp])
    QuasiParticle(weights, f.basis)
end

function majoranas(qp::QuasiParticle)
    return qp + qp', qp - qp'
end

Base.keys(b::FermionBdGBasis) = keys(b.position)
labels(b::FermionBdGBasis) = keys(b).values
Base.keys(qp::QuasiParticle) = keys(qp.weights)
labels(qp::QuasiParticle) = keys(qp).values
basis(qp::QuasiParticle) = qp.basis
function _left_half_labels(basis::FermionBdGBasis)
    N = nbr_of_fermions(basis)
    labels(basis)[1:Int(ceil(N / 2))]
end
function majorana_polarization(f::QuasiParticle, labels=_left_half_labels(basis(f)))
    md1, md2 = majorana_densities(f, labels)
    (md1 - md2) / (md1 + md2)
end

function majorana_coefficients(f::QuasiParticle, labels=labels(basis(f)))
    x = map(l -> f[l, :h] + f[l, :p], labels)
    y = map(l -> 1im * (f[l, :h] - f[l, :p]), labels)
    return Dictionary(labels, x),
    Dictionary(labels, y)
end
function majorana_densities(f::QuasiParticle, labels=_left_half_labels(basis(f)))
    γplus, γminus = majorana_coefficients(f, labels)
    sum(abs2, γplus), sum(abs2, γminus)
end

Base.collect(χ::QuasiParticle) = vcat([χ[key, :h] for key in keys(basis(χ))], [χ[key, :p] for key in keys(basis(χ))])

"""
    one_particle_density_matrix(χ::QuasiParticle{T})

    Gives the one_particle_density_matrix for the state with χ as it's ground state
"""
function one_particle_density_matrix(χ::QuasiParticle{T}) where {T}
    # N = nbr_of_fermions(basis(χ))
    U = collect(χ)
    conj(U) * transpose(U)
end
function one_particle_density_matrix(χs::AbstractVector{<:QuasiParticle})
    sum(one_particle_density_matrix, χs)
end

function one_particle_density_matrix(U::AbstractMatrix{T}) where {T}
    N = div(size(U, 1), 2)
    ρ = zeros(T, 2N, 2N)
    for i in 1:N
        ρ += conj(U[:, i]) * transpose(U[:, i])
    end
    return ρ
end
const DEFAULT_PH_CUTOFF = 1e-12
enforce_ph_symmetry(F::Eigen; cutoff=DEFAULT_PH_CUTOFF) = enforce_ph_symmetry(F.values, F.vectors; cutoff)
function quasiparticle_adjoint(v::AbstractVector)
    Base.require_one_based_indexing(v)
    N = div(length(v), 2)
    out = similar(v)
    for i in 1:N
        out[i] = conj(v[i+N])
        out[i+N] = conj(v[i])
    end
    return out
end
energysort(e) = e #(sign(e), abs(e))
quasiparticle_adjoint_index(n, N) = 2N + 1 - n #n+N
function enforce_ph_symmetry(es, ops; cutoff=DEFAULT_PH_CUTOFF)
    p = sortperm(es, by=energysort)
    es = es[p]
    ops = complex(ops[:, p])
    N = div(length(es), 2)
    ph = quasiparticle_adjoint
    for k in Iterators.take(eachindex(es), N)
        k2 = quasiparticle_adjoint_index(k, N)
        if es[k] > cutoff
            @warn isapprox(es[k], -es[k2], atol=cutoff) "$(es[k]) != $(-es[k2])"
        end
        op = ops[:, k]
        op_ph = ph(op)
        if abs(dot(op_ph, op)) < cutoff #op is not a majorana
            ops[:, k2] = op_ph
        else #it is at least a little bit of majorana
            op2 = ops[:, k2]
            majplus = begin
                v = ph(op) + op
                if norm(v) > cutoff
                    v
                else
                    1im * (ph(op) - op)
                end
            end
            majminus = begin
                v = ph(op2) - op2
                if norm(v) > cutoff
                    v
                else
                    1im * (ph(op2) + op2)
                end
            end
            majs = [majplus majminus]
            @assert all(norm.(eachcol(majs)) .> cutoff)
            X = cholesky(Hermitian(majs' * majs))
            newmajs = majs * inv(X.U)
            @assert newmajs' * newmajs ≈ I
            o1 = (newmajs[:, 1] + 1 * newmajs[:, 2])
            o2 = (newmajs[:, 1] - 1 * newmajs[:, 2])
            normalize!(o1)
            normalize!(o2)
            if abs(dot(o1, op)) > abs(dot(o2, op))
                ops[:, k] = o1
                ops[:, k2] = o2
            else
                ops[:, k] = o2
                ops[:, k2] = o1
            end
        end
    end
    es, ops
end

function check_ph_symmetry(es, ops; cutoff=DEFAULT_PH_CUTOFF)
    N = div(length(es), 2)
    p = sortperm(es, by=energysort)
    inds = Iterators.take(eachindex(es), N)
    all(abs(es[p[i]] + es[p[quasiparticle_adjoint_index(i, N)]]) < cutoff for i in inds) || return false
    all(quasiparticle_adjoint(ops[:, p[i]]) ≈ ops[:, p[quasiparticle_adjoint_index(i, N)]] for i in inds) || return false
    ops' * ops ≈ I || return false
end


function one_particle_density_matrix(ρ::AbstractMatrix{T}, b::FermionBasis) where {T}
    N = nbr_of_fermions(b)
    hoppings = zeros(T, N, N)
    pairings = zeros(T, N, N)
    hoppings2 = zeros(T, N, N)
    pairings2 = zeros(T, N, N)
    for (n, (f1, f2)) in enumerate(Base.product(b.dict, b.dict))
        pairings[n] += tr(ρ * f1 * f2)
        pairings2[n] += tr(ρ * f1' * f2')# -conj(pairings[n])#
        hoppings[n] += tr(ρ * f1' * f2)
        hoppings2[n] += tr(ρ * f1 * f2')
    end
    return [hoppings pairings2; pairings hoppings2]
end

function Base.:*(f1::QuasiParticle, f2::QuasiParticle; kwargs...)
    b = basis(f1)
    @assert b == basis(f2)
    sum(*(BdGFermion(first(l1), b, w1, last(l1) == :h), BdGFermion(first(l2), b, w2, last(l2) == :h); kwargs...) for ((l1, w1), (l2, w2)) in Base.product(pairs(f1.weights), pairs(f2.weights)))
end
function Base.:*(f1::QuasiParticle, f2::BdGFermion; kwargs...)
    b = basis(f1)
    @assert b == basis(f2)
    sum(*(BdGFermion(first(l1), b, w1, last(l1) == :h), f2; kwargs...) for (l1, w1) in pairs(f1.weights))
end
function Base.:*(f1::BdGFermion, f2::QuasiParticle; kwargs...)
    b = basis(f1)
    @assert b == basis(f2)
    sum(*(f1, BdGFermion(first(l2), b, w2, last(l2) == :h); kwargs...) for (l2, w2) in pairs(f2.weights))
end

adj_key(key) = last(key) == :h ? (first(key), :p) : (first(key), :h)
function Base.adjoint(f::QuasiParticle)
    newkeys = map(adj_key, keys(f.weights)).values
    QuasiParticle(Dictionary(newkeys, conj.(f.weights.values)), basis(f))
end
function Base.:+(f1::QuasiParticle, f2::QuasiParticle)
    @assert basis(f1) == basis(f2)
    allkeys = merge(keys(f1.weights), keys(f2.weights))
    newweights = [get(f1.weights, key, false) + get(f2.weights, key, false) for key in allkeys]
    newdict = Dictionary(allkeys, newweights)
    QuasiParticle(newdict, basis(f1))
end
function Base.:-(f1::QuasiParticle, f2::QuasiParticle)
    @assert basis(f1) == basis(f2)
    allkeys = merge(keys(f1.weights), keys(f2.weights))
    newweights = [get(f1.weights, key, false) - get(f2.weights, key, false) for key in allkeys]
    newdict = Dictionary(allkeys, newweights)
    QuasiParticle(newdict, basis(f1))
end
Base.:*(x::Number, f::QuasiParticle) = QuasiParticle(map(Base.Fix1(*, x), f.weights), basis(f))
Base.:*(f::QuasiParticle, x::Number) = QuasiParticle(map(Base.Fix2(*, x), f.weights), basis(f))
Base.:/(f::QuasiParticle, x::Number) = QuasiParticle(map(Base.Fix2(/, x), f.weights), basis(f))

Base.promote_rule(::Type{<:BdGFermion}, ::Type{QuasiParticle{T,M,S}}) where {T,M,S} = QuasiParticle{T,M,S}
Base.convert(::Type{<:QuasiParticle}, f::BdGFermion) = QuasiParticle(f)
Base.:+(f1::AbstractBdGFermion, f2::AbstractBdGFermion) = +(promote(f1, f2)...)
Base.:-(f1::AbstractBdGFermion, f2::AbstractBdGFermion) = -(promote(f1, f2)...)
Base.:+(f1::BdGFermion, f2::BdGFermion) = QuasiParticle(f1) + QuasiParticle(f2)
Base.:-(f1::BdGFermion, f2::BdGFermion) = QuasiParticle(f1) - QuasiParticle(f2)
rep(qp::QuasiParticle) = sum((lw) -> rep(BdGFermion(first(first(lw)), basis(qp), last(lw), last(first(lw)) == :h)), pairs(qp.weights))

function many_body_fermion(f::BdGFermion, basis::FermionBasis)
    if f.hole
        return f.amp * basis[f.id]
    else
        return f.amp * basis[f.id]'
    end
end
function many_body_fermion(qp::QuasiParticle, basis::FermionBasis)
    mbferm((l, w)) = last(l) == :h ? w * basis[first(l)] : w * basis[first(l)]'
    sum(mbferm, pairs(qp.weights))
end

function ground_state_parity(vals, vecs)
    p = sortperm(vals, by=energysort)
    N = div(length(vals), 2)
    pinds = p[[1:N; quasiparticle_adjoint_index.(1:N, N)]]
    sign(det(vecs[:, pinds]))
end

function isantisymmetric(A::AbstractMatrix)
    indsm, indsn = axes(A)
    if indsm != indsn
        return false
    end
    for i = first(indsn):last(indsn), j = (i):last(indsn)
        if A[i, j] != -A[j, i]
            return false
        end
    end
    return true
end
function isbdgmatrix(H, Δ, Hd, Δd)
    indsm, indsn = axes(H)
    if indsm != indsn
        return false
    end
    for i = first(indsn):last(indsn), j = (i):last(indsn)
        if H[i, j] != conj(H[j, i])
            return false
        end
        if H[i, j] != -conj(Hd[i, j])
            return false
        end
        if Δ[i, j] != -conj(Δd[i, j])
            return false
        end
    end
    return true
end

struct BdGMatrix{T,SH,SΔ} <: AbstractMatrix{T}
    # [H Δ; -conj(Δ) -conj(H)]
    H::SH # Hermitian
    Δ::SΔ # Antisymmetric
    function BdGMatrix(H::SH, Δ::SΔ; check=true) where {SH,SΔ}
        @assert size(H) == size(Δ)
        if check
            ishermitian(H) || throw(ArgumentError("H must be hermitian"))
            isantisymmetric(Δ) || throw(ArgumentError("Δ must be antisymmetric"))
        end
        T = promote_type(eltype(H), eltype(Δ))
        # S = promote_type(typeof(H), typeof(Δ))
        new{T,SH,SΔ}(H, Δ)
    end
end
# BdGMatrix(H::Hermitian, Δ; check=true) = BdGMatrix((H), Δ; check)  
function Base.getindex(A::BdGMatrix, i, j)
    N = size(A.H, 1)
    i <= N && j <= N && return A.H[i, j]
    i <= N && j > N && return A.Δ[i, j-N]
    i > N && j <= N && return -conj(A.Δ[i-N, j])
    i > N && j > N && return -conj(A.H[i-N, j-N])
end
Base.:*(x::Real, A::BdGMatrix) = BdGMatrix(x * A.H, x * A.Δ)
Base.:*(A::BdGMatrix, x::Real) = BdGMatrix(A.H * x, A.Δ * x)
Base.:+(A::BdGMatrix, B::BdGMatrix) = BdGMatrix(A.H + B.H, A.Δ + B.Δ)
Base.:-(A::BdGMatrix, B::BdGMatrix) = BdGMatrix(A.H - B.H, A.Δ - B.Δ)

Base.Matrix(A::BdGMatrix) = [A.H A.Δ; -conj(A.Δ) -conj(A.H)]
function BdGMatrix(A::AbstractMatrix; check=true)
    N = div(size(A, 1), 2)
    inds1, inds2 = axes(A)
    H = @views A[inds1[1:N], inds2[1:N]]
    Δ = @views A[inds1[1:N], inds2[N+1:2N]]
    Hd = @views A[inds1[N+1:2N], inds2[N+1:2N]]
    Δd = @views A[inds1[N+1:2N], inds2[1:N]]
    if check
        isbdgmatrix(H, Δ, Hd, Δd) || throw(ArgumentError("A must be a BdGMatrix"))
    end
    BdGMatrix(H, Δ; check)
end
Base.size(A::BdGMatrix, i) = 2size(A.H, i)
Base.size(A::BdGMatrix) = 2 .* size(A.H)

@static if VERSION ≥ v"1.10-"
    function LinearAlgebra.hermitianpart!(m::BdGMatrix)
        m.Δ .-= transpose(m.Δ)
        rdiv!(m.Δ, 2)
        BdGMatrix(hermitianpart!(m.H), m.Δ)
    end
    function LinearAlgebra.hermitianpart(m::BdGMatrix)
        Δ = similar(m.Δ)
        tΔ = transpose(m.Δ)
        @. Δ = (m.Δ - tΔ) / 2
        BdGMatrix(hermitianpart(m.H), Δ)
    end
end

function bdg_to_skew(A::BdGMatrix)
    H = A.H
    Δ = A.Δ
    N = div(size(A, 1), 2)
    A = zeros(real(eltype(A)), 2N, 2N)
    for i in 1:N, j in 1:N
        A[i, j] = imag(-H[i, j] - Δ[i, j])
        A[i, j+N] = real(H[i, j] - Δ[i, j])
        A[j+N, i] = -A[i, j+N]
        A[i+N, j+N] = imag(-H[i, j] + Δ[i, j])
    end
    SkewHermitian(A)
end

bdg_to_skew(bdgham::AbstractMatrix) = bdg_to_skew(BdGMatrix(bdgham))

function skew_qp_adjoint(v)
    vout = similar(v)
    N = div(length(v), 2)
    for j in 1:N
        vout[j] = v[j]
        vout[j+N] = -v[j+N]
    end
    return vout
end
function skew_eigen_to_bdg(es, ops)
    T = complex(eltype(ops))
    N = div(length(es), 2)
    pair_itr = collect(Iterators.partition(es, 2)) #take each eigenpair
    p = sortperm(pair_itr, by=Base.Fix1(-, 0) ∘ abs ∘ first) #permutation to sort the pairs
    internal_p = map(pair -> sortperm(pair), pair_itr[p]) #permutation within each pair
    pinds = vcat([2p - 1 + pp[1] - 1 for (p, pp) in zip(p, internal_p)],
        reverse([2p - 1 + pp[2] - 1 for (p, pp) in zip(p, internal_p)])) #puts all negative energies first and positive energies at the adjoint indices
    ops2 = zeros(T, 2N, 2N)
    for i in 1:N, j in 1:2N
        ops2[i, j] = (ops[i, j] - 1im * ops[i+N, j]) / sqrt(2)
        ops2[i+N, j] = (ops[i, j] + 1im * ops[i+N, j]) / sqrt(2)
    end
    return es[pinds], ops2[:, pinds]
end

abstract type AbstractBdGEigenAlg end

struct SkewEigenAlg{T<:Number} <: AbstractBdGEigenAlg
    cutoff::T # tolerance for particle-hole symmetry
end
struct NormalEigenAlg{T<:Number} <: AbstractBdGEigenAlg
    cutoff::T # tolerance for particle-hole symmetry
end
NormalEigenAlg() = NormalEigenAlg(DEFAULT_PH_CUTOFF)
SkewEigenAlg() = SkewEigenAlg(DEFAULT_PH_CUTOFF)

function diagonalize(A::BdGMatrix, alg::NormalEigenAlg)
    QuantumDots.enforce_ph_symmetry(eigen(Matrix(A)), cutoff=alg.cutoff)
end
diagonalize(A::BdGMatrix) = diagonalize(A, NormalEigenAlg(DEFAULT_PH_CUTOFF))
function diagonalize(A::AbstractMatrix, alg::SkewEigenAlg)
    es, ops = eigen(bdg_to_skew(A))
    enforce_ph_symmetry(skew_eigen_to_bdg(imag.(es), ops)...; cutoff=alg.cutoff)
end