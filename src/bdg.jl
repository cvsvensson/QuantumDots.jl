struct QuasiParticle{T,M,L} <: AbstractBdGFermion
    weights::Dictionary{Tuple{L,Symbol},T}
    basis::FermionBdGBasis{M,L}
end
function QuasiParticle(v::AbstractVector{T}, basis::FermionBdGBasis{M,L}) where {T,M,L}
    holelabels = map(k -> (k, :h), keys(basis.position).values)
    particlelabels = map(k -> (k, :p), keys(basis.position).values)
    weights = Dictionary(vcat(holelabels, particlelabels), collect(v))
    QuasiParticle{T,M,L}(weights, basis)
end
Base.getindex(qp::QuasiParticle, i) = getindex(qp.weights, i)
Base.getindex(qp::QuasiParticle, i...) = getindex(qp.weights, i)

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

function majorana_wavefunctions(f::QuasiParticle, labels=labels(basis(f)))
    xylabels = [map(l -> (l, :x), labels); map(l -> (l, :y), labels)]
    xplus = map(l -> real(f[l, :h] + f[l, :p]), labels)
    xminus = map(l -> imag(f[l, :h] + f[l, :p]), labels)
    yplus = map(l -> imag(f[l, :h] - f[l, :p]), labels)
    yminus = map(l -> real(f[l, :h] - f[l, :p]), labels)
    return Dictionary(xylabels, [xplus; yplus]),
    Dictionary(xylabels, [xminus; yminus])
end
function majorana_densities(f::QuasiParticle, labels=_left_half_labels(basis(f)))
    γplus, γminus = majorana_wavefunctions(f, labels)
    sum(abs2, γplus), sum(abs2, γminus)
end

"""
    one_particle_density_matrix(χ::QuasiParticle{T})

    Gives the one_particle_density_matrix for the state with χ as it's ground state
"""
function one_particle_density_matrix(χ::QuasiParticle{T}) where {T}
    N = nbr_of_fermions(basis(χ))
    U = χ.weights.values
    U * transpose(U)
end
function one_particle_density_matrix(χs::AbstractVector{<:QuasiParticle})
    sum(one_particle_density_matrix, χs)
end

function one_particle_density_matrix(U::AbstractMatrix{T}) where {T}
    N = div(size(U, 1), 2)
    ρ = zeros(T, 2N, 2N)
    for i in 1:N
        ρ += U[:, i] * transpose(U[:, i])
    end
    return ρ
end
enforce_ph_symmetry(F::Eigen) = enforce_ph_symmetry(F.values, F.vectors)
quasiparticle_adjoint(v, N=div(length(v), 2)) = [conj(v[N+1:2N]); conj(v[1:N])]
energysort(e) = e #(sign(e), abs(e))
quasiparticle_adjoint_index(n, N) = 2N + 1 - n #n+N
function enforce_ph_symmetry(es, ops; cutoff=1e-12)
    p = sortperm(es, by=energysort)
    es = es[p]
    ops = ops[:, p]
    N = div(length(es), 2)
    ph = quasiparticle_adjoint
    for k in Iterators.take(eachindex(es), N)
        k2 = quasiparticle_adjoint_index(k, N)
        if es[k] > cutoff
            @warn isapprox(es[k], -es[k2], atol=cutoff) "$(es[k]) != $(-es[k2])"
        end
        op = ops[:, k]
        op2 = ph(op)
        if abs(dot(op2, op)) < cutoff #op is not a majorana
            ops[:, k2] = op2
        else
            majplus = ph(op) + op + (ph(ops[:, k2]) + ops[:, k2]) / 2
            normalize!(majplus)
            majminus = ph(op) - op + (ph(ops[:, k2]) - ops[:, k2]) / 2
            normalize!(majminus)
            o1 = (majplus + 1 * majminus) / sqrt(2)
            o2 = (majplus - 1 * majminus) / sqrt(2)
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

function check_ph_symmetry(es, ops; cutoff=1e-11)
    N = div(length(es), 2)
    p = sortperm(es, by=energysort)
    inds = Iterators.take(eachindex(es), N)
    all(abs(es[p[i]] + es[p[quasiparticle_adjoint_index(i, N)]]) < cutoff for i in inds) &&
        all(quasiparticle_adjoint(ops[:, p[i]]) ≈ ops[:, p[quasiparticle_adjoint_index(i, N)]] for i in inds) &&
        ops' * ops ≈ I
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
Base.adjoint(f::QuasiParticle) = QuasiParticle(Dictionary(keys(f.weights).values, quasiparticle_adjoint(f.weights.values, nbr_of_fermions(basis(f)))), basis(f))

function Base.:+(f1::QuasiParticle, f2::QuasiParticle)
    @assert basis(f1) == basis(f2)
    QuasiParticle(map(+, f1.weights, f2.weights), basis(f1))
end
function Base.:-(f1::QuasiParticle, f2::QuasiParticle)
    @assert basis(f1) == basis(f2)
    QuasiParticle(map(-, f1.weights, f2.weights), basis(f1))
end
Base.:*(x::Number, f::QuasiParticle) = QuasiParticle(map(Base.Fix1(*, x), f.weights), basis(f))
Base.:*(f::QuasiParticle, x::Number) = QuasiParticle(map(Base.Fix2(*, x), f.weights), basis(f))
Base.:/(f::QuasiParticle, x::Number) = QuasiParticle(map(Base.Fix2(/, x), f.weights), basis(f))

QuasiParticle(f::BdGFermion) = QuasiParticle(rep(f), f.basis)
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


# function majorana_bdg_transform(N)
#     i = Matrix(I / sqrt(2), N, N)
#     [i i; 1im*i -1im*i]
# end
# function bdg_to_skew2(bdgham::AbstractMatrix; U=majorana_bdg_transform(div(size(bdgham, 1), 2)))
#     SkewHermitian(-imag(U * bdgham * U'))
# end
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
    indsm, indsn = axes(A)
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

struct BdGMatrix{T,S} <: AbstractMatrix{T}
    # [H Δ; -conj(Δ) -conj(H)]
    H::S # Hermitian
    Δ::S # Antisymmetric
    function BdGMatrix(H::S1, Δ::S2) where {S1,S2}
        @assert size(H) == size(Δ)
        ishermitian(H) || throw(ArgumentError("H must be hermitian"))
        isantisymmetric(Δ) || throw(ArgumentError("Δ must be antisymmetric"))
        H2, Δ2 = promote(H, Δ)
        new{eltype(H2),typeof(H2)}(H2, Δ2)
    end
end
Base.Matrix(A::BdGMatrix) = [A.H A.Δ; -conj(A.Δ) -conj(A.H)]
function BdGMatrix(A::AbstractMatrix; view=false)
    N = div(size(bdgham, 1), 2)
    inds1, inds2 = axes(A)
    if view
        H = @views A[inds1[1:N], inds2[1:N]]
        Δ = @views A[inds1[1:N], inds2[N+1:2N]]
        Hd = @views A[inds1[N+1:2N], inds2[N+1:2N]]
        Δd = @views A[inds1[N+1:2N], inds2[1:N]]
    else
        H = A[inds1[1:N], inds2[1:N]]
        Δ = A[inds1[1:N], inds2[N+1:2N]]
        Hd = A[inds1[N+1:2N], inds2[N+1:2N]]
        Δd = A[inds1[N+1:2N], inds2[1:N]]
    end
    isbdgmatrix(H, Δ, Hd, Δd) || throw(ArgumentError("A must be a BdGMatrix"))
    BdGMatrix(H, Δ)
end

function bdg_to_skew(bdgham::AbstractMatrix{T}) where {T}
    N = div(size(bdgham, 1), 2)
    A = zeros(real(T), 2N, 2N)
    inds1 = axes(bdgham, 1)
    inds2 = axes(bdgham, 2)
    a = @views bdgham[inds1[1:N], inds2[1:N]]
    b = @views bdgham[inds1[1:N], inds2[N+1:2N]]
    d = -a
    c = b'
    # c = @views bdgham[inds1[N+1:2N], inds2[1:N]]
    # d = @views bdgham[inds1[N+1:2N], inds2[N+1:2N]]
    @. A[1:N, 1:N] = -imag(a + b + c + d) / 2
    # @. A[1:N, 1:N] = -imag(b + c) ./ 2
    @. A[1:N, N+1:2N] = -imag(1im * (b + d - a - c)) / 2
    @. A[N+1:2N, 1:N] = -imag(1im * (a - c + b - d)) / 2
    @. A[N+1:2N, N+1:2N] = -imag(a - b - c + d) / 2
    # @. A[N+1:2N, N+1:2N] = imag(-b - c) / 2
    @assert norm(A + A') < 1e-12
    skewhermitian!(A)
    # SkewHermitian(A)
end
function skew_eigen_to_bdg(es, ops)#; U=majorana_bdg_transform(div(length(es), 2)))
    T = complex(eltype(ops))
    phases = Diagonal([iseven(k) ? one(T) * 1im : one(T) for k in 1:length(es)])
    p = sortperm(es, by=energysort)

    N = div(size(ops, 1), 2)
    inds1 = axes(ops, 1)
    inds2 = axes(ops, 2)
    a = @views ops[inds1[1:N], inds2[1:N]]
    b = @views ops[inds1[1:N], inds2[N+1:2N]]
    c = @views ops[inds1[N+1:2N], inds2[1:N]]
    d = @views ops[inds1[N+1:2N], inds2[N+1:2N]]
    ops2 = similar(ops, T)
    @. ops2[inds1[1:N], inds2[1:N]] = a - 1im * c
    @. ops2[inds1[1:N], inds2[N+1:2N]] = b - 1im * d
    @. ops2[inds1[N+1:2N], inds2[1:N]] = a + 1im * c
    @. ops2[inds1[N+1:2N], inds2[N+1:2N]] = b + 1im * d
    es[p], (ops2*phases)[:, p]
    # es[p], (U'*ops*d)[:, p]
end

function skeweigen(bdgham)
    N = div(size(bdgham, 1), 2)
    # U = majorana_bdg_transform(N)
    A = bdg_to_skew(bdgham)#; U)
    esA, opsA = eigen(A)
    skew_eigen_to_bdg(imag.(esA), opsA)
end