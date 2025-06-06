"""
    struct QuasiParticle{T,M,L} <: AbstractBdGFermion

The `QuasiParticle` struct represents a quasi-particle in the context of a BdG (Bogoliubov-de Gennes) fermion system. It is a linear combination of basis BdG fermions, and is defined by a set of weights and a basis.
"""
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

"""
    QuasiParticle(f::BdGFermion)

Constructs a `QuasiParticle` object from a `BdGFermion` object.
"""
function QuasiParticle(f::BdGFermion)
    label = (f.id, f.hole ? :h : :p)
    weights = Dictionary([label], [f.amp])
    QuasiParticle(weights, f.basis)
end

function majoranas(qp::QuasiParticle)
    return qp + qp', qp - qp'
end

Base.keys(b::FermionBdGBasis) = keys(b.position)
Base.keys(qp::QuasiParticle) = keys(qp.weights)
basis(qp::QuasiParticle) = qp.basis
function _left_half_labels(basis::FermionBdGBasis)
    N = nbr_of_modes(basis)
    collect(keys(basis))[1:Int(ceil(N / 2))]
end
function majorana_polarization(f::QuasiParticle, labels=_left_half_labels(basis(f)))
    md1, md2 = majorana_densities(f, labels)
    (md1 - md2) / (md1 + md2)
end

"""
    majorana_coefficients(f::QuasiParticle, labels=collect(keys((basis(f))))

Compute the Majorana coefficients for a given `QuasiParticle` object `f`. Returns two dictionaries, for the two types of Majorana operators.
"""
function majorana_coefficients(f::QuasiParticle, labels=collect(keys(basis(f))))
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

    Return the one_particle_density_matrix for the state with χ occupied as its ground state.
"""
function one_particle_density_matrix(χ::QuasiParticle{T}) where {T}
    U = collect(χ)
    conj(U) * transpose(U)
end

"""
    one_particle_density_matrix(χs::AbstractVector{<:QuasiParticle})

Return the one_particle_density_matrix for the state with all χs occupied as its ground state.
"""
function one_particle_density_matrix(χs::AbstractVector{<:QuasiParticle})
    sum(one_particle_density_matrix, χs)
end

"""
    one_particle_density_matrix(U::AbstractMatrix{T}) where {T}

Compute the one-particle density matrix for the vacuum of a BdG system diagonalized by `U`.
"""
function one_particle_density_matrix(U::AbstractMatrix{T}) where {T}
    N = div(size(U, 1), 2)
    ρ = zeros(T, 2N, 2N)
    for i in 1:N
        ρ += conj(U[:, i]) * transpose(U[:, i])
    end
    return ρ
end
const DEFAULT_PH_CUTOFF = 1e-12
function enforce_ph_symmetry(F::Eigen; cutoff=DEFAULT_PH_CUTOFF)
    if isreal(F.values)
        enforce_ph_symmetry(real(F.values), F.vectors; cutoff)
    else
        throw(ArgumentError("Eigenvalues must be real"))
    end
end
"""
    quasiparticle_adjoint(v::AbstractVector)

Compute the adjoint of a quasiparticle represented by the weights in `v`. The adjoint is computed by swapping the hole and particle parts of the vector and taking the complex conjugate of each element.
"""
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
function enforce_ph_symmetry(_es, _ops; cutoff=DEFAULT_PH_CUTOFF)
    p = sortperm(_es, by=energysort)
    es = _es[p]
    ops = complex(_ops[:, p])
    N = div(length(es), 2)
    ph = quasiparticle_adjoint
    for k in Iterators.take(eachindex(es), N)
        k2 = quasiparticle_adjoint_index(k, N)
        if es[k] > cutoff && isapprox(es[k], -es[k2], atol=cutoff)
            @warn "es[k] = $(es[k]) != $(-es[k2]) = -es[k_adj]"
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
                if norm(v) > cutoff && abs(dot(v, majplus)) < norm(majplus)^2
                    v
                else
                    1im * (ph(op2) + op2)
                end
            end
            majs = [majplus majminus]
            if !all(norm.(eachcol(majs)) .> cutoff)
                @warn "Norm of majoranas = $(norm.(eachcol(majs)))"
                @debug "Norm of majoranas is small. Majoranas:" majs
            end
            HM = Hermitian(majs' * majs)
            X = try
                cholesky(HM)
            catch
                vals = eigvals(HM)
                @warn "Cholesky failed, matrix is not positive definite? eigenvals = $vals. Adding $cutoff * I"
                @debug "Cholesky failed. Input:" _es _ops
                cholesky(HM + cutoff * I)
            end
            newmajs = majs * inv(X.U)
            if !(newmajs' * newmajs ≈ I)
                @warn "New majoranas are not orthogonal? $(norm(newmajs' * newmajs - I))"
                @debug "New majoranas are not orthogonal? New majoranas:" newmajs
            end
            o1 = (newmajs[:, 1] + newmajs[:, 2])
            o2 = (newmajs[:, 1] - newmajs[:, 2])
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


"""
    one_particle_density_matrix(ρ::AbstractMatrix, b::FermionBasis, labels=keys(b))

Compute the one-particle density matrix for a given density matrix `ρ` in the many body fermion basis `b`.
"""
function one_particle_density_matrix(ρ::AbstractMatrix{T}, b::FermionBasis, labels=keys(b)) where {T}
    N = length(labels)
    hoppings = zeros(T, N, N)
    pairings = zeros(T, N, N)
    for (n, (l1, l2)) in enumerate(Base.product(labels, labels))
        f1 = b[l1]
        f2 = b[l2]
        pairings[n] += tr(ρ * f1 * f2)
        hoppings[n] += tr(ρ * f1' * f2)
    end
    pairings = (pairings - transpose(pairings)) / 2
    hoppings = hermitianpart!(hoppings)
    return [hoppings -conj(pairings); pairings (I*tr(ρ)-conj(hoppings))]
end

"""
    *(f1::QuasiParticle, f2::QuasiParticle; kwargs...)

Return the BdG matrix of the product of quasiparticles `f1` and `f2`.
"""
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

"""
    many_body_fermion(f::BdGFermion, basis::FermionBasis)

Return the representation of `f` in the many-body fermion basis `basis`.
"""
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

"""
    struct BdGMatrix <: AbstractMatrix

BdGMatrix represents a Bogoliubov-de Gennes (BdG) matrix, which is a block matrix used to describe non-interacting superconducting systems. It consists of four blocks: H, Δ, -conj(Δ), and -conj(H), where H is a Hermitian matrix and Δ is an antisymmetric matrix.

# Fields
- `H`: The Hermitian block of the BdG matrix.
- `Δ`: The antisymmetric block of the BdG matrix.
"""
struct BdGMatrix{T,SH,SΔ} <: AbstractMatrix{T}
    H::SH # Hermitian
    Δ::SΔ # Antisymmetric
    function BdGMatrix(H::SH, Δ::SΔ; check=true) where {SH,SΔ}
        @assert size(H) == size(Δ)
        if check
            ishermitian(H) || throw(ArgumentError("H must be hermitian"))
            isantisymmetric(Δ) || throw(ArgumentError("Δ must be antisymmetric"))
        end
        T = promote_type(eltype(H), eltype(Δ))
        new{T,SH,SΔ}(Hermitian(H), Δ)
    end
end
function Base.getindex(A::BdGMatrix, i::Integer, j::Integer)
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

Base.hvcat(A::BdGMatrix) = [A.H A.Δ; -conj(A.Δ) -conj(A.H)]
function Base.Matrix(A::BdGMatrix)
    n, m = size(A.H)
    T = promote_type(eltype(A.H), eltype(A.Δ))
    out = Matrix{T}(undef, 2n, 2n)
    for j in axes(A.H, 2), i in axes(A.H, 1)
        out[i, j] = A.H[i, j]
        out[i, j+n] = A.Δ[i, j]
        out[i+n, j] = -conj(A.Δ[i, j])
        out[i+n, j+n] = -conj(A.H[i, j])
    end
    return out
end
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
    BdGMatrix(hermitianpart(H), (Δ - transpose(Δ)) / 2; check=false)
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

"""
    bdg_to_skew(B::BdGMatrix; check=true)

Convert a BdGMatrix to a skew-Hermitian matrix. If `check` is true, it checks that the result is skew-Hermitian.
"""
function bdg_to_skew(B::BdGMatrix; check=true)
    bdg_to_skew(B.H, B.Δ; check)
end
function bdg_to_skew(H::AbstractMatrix, Δ::AbstractMatrix; check=true)
    N = size(H, 1)
    T = real(promote_type(eltype(H), eltype(Δ)))
    A = zeros(T, 2N, 2N)
    for i in 1:N, j in 1:N
        A[i, j] = imag(H[i, j] + Δ[i, j])
        A[i+N, j] = real(H[i, j] + Δ[i, j])
        A[j, i+N] = -A[i+N, j]
        A[i+N, j+N] = imag(H[i, j] - Δ[i, j])
    end
    if check
        return SkewHermitian(A)
    else
        return skewhermitian!(A)#SkewHermitian{eltype(A),typeof(A)}(A)
    end
end
bdg_to_skew(bdgham::AbstractMatrix; check=true) = bdg_to_skew(BdGMatrix(bdgham; check); check)

"""
    skew_to_bdg(A::AbstractMatrix)

Convert a skew-symmetric matrix `A` to a BdGMatrix.
"""
function skew_to_bdg(A::AbstractMatrix)
    BdGMatrix(_skew_to_bdg(A)...)
end
function _skew_to_bdg(A::AbstractMatrix)
    N = div(size(A, 1), 2)
    T = complex(eltype(A))
    H = zeros(T, N, N)
    Δ = zeros(T, N, N)
    for i in 1:N, j in i:N
        H[i, j] = (A[i+N, j] - A[i, j+N] + 1im * (A[i, j] + A[i+N, j+N])) / 2
        H[j, i] = conj(H[i, j])
        Δ[i, j] = (A[i+N, j] + A[i, j+N] + 1im * (A[i, j] - A[i+N, j+N])) / 2
        Δ[j, i] = -Δ[i, j]
        if i == j
            Δ[j, j] = 0
        end
    end
    return Hermitian(H), Δ
end

"""
    skew_to_bdg(v::AbstractVector)

Use the same transformation that transforms a skew-symmetric matrix to a BdGMatrix to transform a vector `v`.
"""
function skew_to_bdg(v::AbstractVector)
    N = div(length(v), 2)
    T = complex(eltype(v))
    uv = zeros(T, 2N)
    for i in 1:N
        uv[i] = (v[i] - 1im * v[i+N]) / sqrt(2)
        uv[i+N] = (v[i] + 1im * v[i+N]) / sqrt(2)
    end
    return uv
end

function skew_eigen_to_bdg(_es, ops)
    es = imag(-_es)
    pair_itr = collect(Iterators.partition(es, 2)) #take each eigenpair
    p = sortperm(pair_itr, by=Base.Fix1(-, 0) ∘ abs ∘ first) #permutation to sort the pairs
    internal_p = map(pair -> sortperm(pair), pair_itr[p]) #permutation within each pair
    pinds = vcat([2p - 1 + pp[1] - 1 for (p, pp) in zip(p, internal_p)],
        reverse([2p - 1 + pp[2] - 1 for (p, pp) in zip(p, internal_p)])) #puts all negative energies first and positive energies at the adjoint indices
    H = stack(skew_to_bdg, eachcol(ops))
    return es[pinds], H[:, pinds]
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
# diagonalize(A::BdGMatrix) = diagonalize(A, SkewEigenAlg(DEFAULT_PH_CUTOFF))
function diagonalize(A::AbstractMatrix, alg::SkewEigenAlg)
    es, ops = eigen(bdg_to_skew(A))
    enforce_ph_symmetry(skew_eigen_to_bdg(es, ops)...; cutoff=alg.cutoff)
end


function many_body_density_matrix_exp(G, c=FermionBasis(1:div(size(G, 1), 2), qn=parity); alg=SkewEigenAlg())
    G = remove_trace(G)
    vals, vecs = diagonalize(BdGMatrix(G; check=false), alg)
    clamp_val(e) = clamp(e, -1 / 2 + eps(e), 1 / 2 - eps(e))
    f(e) = log((e + 1 / 2) / (1 / 2 - e))
    vals2 = map(f ∘ clamp_val, vals[1:div(length(vals), 2)])
    H = vecs * Diagonal(vcat(vals2, -reverse(vals2))) * vecs'
    N = length(vals2)
    _H = Hermitian(H[1:N, 1:N])
    Δ = H[1:N, N+1:2N]
    Δ = (Δ - transpose(Δ)) / 2
    @assert _H ≈ -transpose(H[N+1:2N, N+1:2N])
    @assert Δ ≈ -transpose(Δ)
    @assert Δ ≈ -conj(H[N+1:2N, 1:N])
    Hmb = Matrix(many_body_hamiltonian(_H, Δ, c))
    rho = exp(Hmb)
    return rho / tr(rho)
end

remove_trace(A) = A - tr(A)I / size(A, 1)
"""
    many_body_density_matrix(G, c=FermionBasis(1:div(size(G, 1), 2), qn=parity); alg=SkewEigenAlg())

Compute the many-body density matrix for a given correlator G. The traceless version of G should be a BdGMatrix. 

See also [`one_particle_density_matrix`](@ref), [`many_body_hamiltonian`](@ref).
"""
function many_body_density_matrix(G, c=FermionBasis(1:div(size(G, 1), 2), qn=parity); alg=SkewEigenAlg())
    G = remove_trace(G)
    vals, vecs = diagonalize(BdGMatrix(G; check=false), alg)
    cbdg = FermionBdGBasis(c)
    qps = map(i -> QuasiParticle(vecs[:, i], cbdg), 1:size(vecs, 2))
    mbqps = map(qp -> many_body_fermion(qp, c), qps)
    rho = prod((I * (1 / 2 - e) + 2e * Matrix(qp' * qp)) for (e, qp) in zip(vals[1:div(length(vals), 2)], mbqps))
    return rho
end
FermionBdGBasis(c::FermionBasis) = FermionBdGBasis(collect(keys(c)))

function many_body_hamiltonian(H::BdGMatrix, c::FermionBasis=FermionBasis(1:size(H.H, 1), qn=parity))
    many_body_hamiltonian(H.H, H.Δ, c)
end

"""
    many_body_hamiltonian(H::AbstractMatrix, Δ::AbstractMatrix, c::FermionBasis=FermionBasis(1:size(H, 1), qn=parity))

Construct the many-body Hamiltonian for a given BdG Hamiltonian consisting of hoppings `H` and pairings `Δ`.
"""
function many_body_hamiltonian(H::AbstractMatrix, Δ::AbstractMatrix, c::FermionBasis=FermionBasis(1:size(H, 1), qn=parity))
    sum((H[i, j] * c[i]' * c[j] - conj(H[i, j]) * c[i] * c[j]') / 2 - (Δ[i, j] * c[i] * c[j] - conj(Δ[i, j]) * c[i]' * c[j]') / 2 for (i, j) in Base.product(keys(c), keys(c)))
end
