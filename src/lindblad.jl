struct Lindblad <: AbstractOpenSolver end

function _default_lindblad_dissipator_params(l::NormalLead)
    T, μ = promote(temperature(l), chemical_potential(l))
    type = eltype(T)
    rate = one(type)
    props = @LVector type (:T, :μ, :rate)
    props.T = T
    props.μ = μ
    props.rate = rate
    return props
end

function update_superop!(superop, op, energies, μ, T, rate, vectorizer, cache, tmp_type)
    # num = promote(μ,T,rate)[1]
    opcache = get_tmp(cache.opcache, tmp_type)
    mulcache = get_tmp(cache.mulcache, tmp_type)
    kroncache = get_tmp(cache.kroncache, tmp_type)
    superopcache = get_tmp(superop, tmp_type)
    ratetransform!(opcache, op, energies, T, μ)
    dissipator!(superopcache, opcache, rate, vectorizer, kroncache, mulcache)
end

struct LindbladOperator{T,V,D,U,H,C,L,LP} <: AbstractOpenSystem
    hamiltonian::H
    unitary::U
    dissipators::D # Diffcaches
    lead_ops::L
    lead_props::LP
    total::T #Diffcache
    vectorizer::V
    cache::C #Has kroncache, mulcache
end

function update_dissipator!(L, label, props, cache, tmp_type)
    superops = L.dissipators[label]
    ops = L.lead_ops[label]
    energies = eigenvaluevector(L.hamiltonian)
    vectorizer = L.vectorizer
    μ = get(props, :μ, L.lead_props[label].μ)
    T = get(props, :T, L.lead_props[label].T)
    rate = get(props, :rate, L.lead_props[label].rate)
    update_dissipator!(superops, ops, energies, μ, T, rate, vectorizer, cache, tmp_type)
end
function update_dissipator!(superops, ops, energies, μ, T, rate, vectorizer, cache, tmp_type)
    update_superop!(superops.in, ops.in, energies, μ, T, rate, vectorizer, cache, tmp_type)
    update_superop!(superops.out, ops.out, energies, -μ, T, rate, vectorizer, cache, tmp_type)
end


_lead_prop(l) = (; μ=l.μ, T=l.T, rate=one(l.T))

function LindbladOperator(hamiltonian::DiagonalizedHamiltonian, leads, vectorizer=default_vectorizer(hamiltonian))
    commutator_hamiltonian = commutator(eigenvalues(hamiltonian), vectorizer)
    unitary = -1im * commutator_hamiltonian
    total = DiffCache(deepcopy(Matrix(unitary)))
    kroncache = deepcopy(total)
    mulcache = DiffCache(complex(Matrix(eigenvalues(hamiltonian))))
    opcache = DiffCache(complex(Matrix(eigenvalues(hamiltonian))))
    dissipators = map(lead -> (; in=deepcopy(total), out=deepcopy(total)), leads)
    cache = (; kroncache, mulcache, opcache)
    props = map(_lead_prop, leads)
    lead_ops = map(lead -> (; in=lead.jump_in, out=lead.jump_out), leads)
    energies = eigenvaluevector(hamiltonian)
    map(label -> update_dissipator!(dissipators[label], lead_ops[label], energies, props[label].μ, props[label].T, props[label].rate, vectorizer, cache, 1.0), keys(leads))
    update_total_operator!(total, unitary, dissipators, props)
    LindbladOperator(hamiltonian, unitary, dissipators, lead_ops, props, total, vectorizer, cache)
end


LindbladOperator(sys::OpenSystem, vectorizer=default_vectorizer(sys)) = LindbladOperator(sys.hamiltonian, sys.leads, vectorizer)

LinearOperator(L::LindbladOperator{<:DiffCache}, p=SciMLBase.NullParameters(); normalizer=false) = MatrixOperator(L, p; normalizer)
function MatrixOperator(L::LindbladOperator, p=SciMLBase.NullParameters(); normalizer)
    # L = deepcopy(_L)
    # update_func! = lindblad_updater!(L; normalizer)
    update_L!(L, nothing, p, nothing)
    A = normalizer ? lindblad_with_normalizer(L.total.du, L.vectorizer) : L.total.du
    MatrixOperator(A)
end

_pairs(p) = pairs(p)
_pairs(::SciMLBase.NullParameters) = pairs(())

LinearAlgebra.mul!(du, L::LindbladOperator, u) = LinearAlgebra.mul!(du, get_tmp(L.total, u), u)
LinearAlgebra.mul!(du, L::LindbladOperator, u, α, β) = LinearAlgebra.mul!(du, get_tmp(L.total, u), u, α, β)
Base.:*(L::LindbladOperator, u) = get_tmp(L.total, u) * u

(L::LindbladOperator)(u, p, t; kwargs...) = (update_L!(L, u, p, t; kwargs...); get_tmp(L.total, mapreduce(collect, vcat, p, init=eltype(L.total.du)[])) * u)
(L::LindbladOperator)(du, u, p, t; kwargs...) = (update_L!(L, u, p, t; kwargs...); mul!(du, get_tmp(L.total, mapreduce(collect, vcat, p, init=eltype(L.total.du)[])), u))
(L::LindbladOperator)(du, u, p, t, α, β; kwargs...) = (update_L!(L, u, p, t; kwargs...); mul!(du, get_tmp(L.total, mapreduce(collect, vcat, p, init=eltype(L.total.du)[])), u, α, β))
SciMLBase.islinear(L::LindbladOperator) = true


function update_L!(L, u, p, t)
    for (label, props) in _pairs(p)
        tmp_type = promote(props...)[1]
        get_tmp(L.total, tmp_type) .-= get_tmp(L.dissipators[label].in, tmp_type) .+ get_tmp(L.dissipators[label].out, tmp_type)
        update_dissipator!(L, label, props, L.cache, tmp_type)
        get_tmp(L.total, tmp_type) .+= get_tmp(L.dissipators[label].in, tmp_type) .+ get_tmp(L.dissipators[label].out, tmp_type)
    end
    return nothing
end

function update_total_operator!(_total, unitary, dissipators, props)
    vprops = mapreduce(collect, vcat, props)
    total = get_tmp(_total, vprops)
    total .= (unitary)
    for d in dissipators
        total .+= get_tmp(d.in, vprops) .+ get_tmp(d.out, vprops)
    end
end

tomatrix(rho::AbstractVector, system::LindbladOperator) = tomatrix(rho, system.vectorizer)
tomatrix(rho::AbstractVector, vectorizer::KronVectorizer) = reshape(rho, vectorizer.size, vectorizer.size)
tomatrix(rho::AbstractVector, vectorizer::KhatriRaoVectorizer) = BlockDiagonal(map((size, inds) -> reshape(rho[inds], size, size), vectorizer.sizes, vectorizer.vectorinds))
internal_rep(rho, system::LindbladOperator) = internal_rep(rho, system.vectorizer)
internal_rep(rho::AbstractMatrix, ::KronVectorizer) = vec(rho)
internal_rep(rho::UniformScaling, v::KronVectorizer) = vec(Diagonal(rho, v.size))
internal_rep(rho::UniformScaling, v::KhatriRaoVectorizer) = vecdp(BlockDiagonal([Diagonal(rho, sz) for sz in v.sizes]))
internal_rep(rho::BlockDiagonal, vectorizer::KhatriRaoVectorizer) = iscompatible(rho, vectorizer) ? vecdp(rho) : internal_rep(Matrix(rho), vectorizer)
iscompatible(rho::BlockDiagonal, vectorizer::KhatriRaoVectorizer) = size.(rho.blocks, 1) == size.(rho.blocks, 2) == vectorizer.sizes
function internal_rep(rho::AbstractMatrix{T}, vectorizer::KhatriRaoVectorizer) where {T}
    inds = vectorizer.inds
    indsv = vectorizer.vectorinds
    v = Vector{T}(undef, vectorizer.cumsumsquared[end])
    for i in eachindex(inds)
        v[indsv[i]] = vec(rho[inds[i], inds[i]])
    end
    return v
end
internal_rep(rho::AbstractVector, ::AbstractVectorizer) = rho

reprlindblad(lm::LM) where {LM<:LinearMap} = "LinearMap{$(eltype(lm))}"
reprlindblad(m::AbstractMatrix) = typeof(m)
reprdissipators(x) = string(typeof(x), ", Labels: ", map(x -> x.label, x)) #Base.repr(x)

#kron(B,A)*vec(rho) ∼ A*rho*B' ∼ 
# A*rho*B = transpose(B)⊗A = kron(transpose(B),A)
# A*rho*transpose(B) = B⊗A = kron(B,A)
# superjump(L) = kron(L,L) - 1/2*(kron(one(L),L'*L) + kron(L'*L,one(L)))

const DENSE_CUTOFF = 16
const KR_LAZY_CUTOFF = 40
dissipator(L, krv::KhatriRaoVectorizer) = sum(krv.sizes) > KR_LAZY_CUTOFF ? khatri_rao_lazy_dissipator(L, krv.sizes) : khatri_rao_dissipator(L, krv.sizes)
commutator(A, krv::KhatriRaoVectorizer) = sum(krv.sizes) > KR_LAZY_CUTOFF ? khatri_rao_lazy_commutator(A, krv.sizes) : khatri_rao_commutator(A, krv.sizes)

function dissipator(L, kv::KronVectorizer)
    D = (conj(L) ⊗ L - 1 / 2 * kronsum(transpose(L' * L), L' * L))
    return kv.size > DENSE_CUTOFF ? D : Matrix(D)
    # return Matrix(D)
end

dissipator!(out, L::AbstractMatrix, rate, kv::KhatriRaoVectorizer, kroncache, mulcache) = khatri_rao_dissipator!(out, L, rate, kv, kroncache, mulcache)


function dissipator!(out, L::AbstractMatrix{T}, rate, kv::KronVectorizer, kroncache, mulcache) where {T}
    kron!(kroncache, transpose(L'), L)
    out .= kroncache
    i = I(kv.size)
    mul!(mulcache, L', L, 1 / 2, 0)
    kron!(kroncache, mulcache, i)
    out .-= kroncache
    kron!(kroncache, i, mulcache)
    out .-= kroncache
    out .*= rate
    #D = (conj(L) ⊗ L - 1 / 2 * kronsum(transpose(L' * L), L' * L))
    #return kv.size > DENSE_CUTOFF ? D : Matrix(D)
    return out
end

commutator(A, ::KronVectorizer) = commutator(A)
commutator(A) = kron(one(A), A) - kron(transpose(A), one(A))
measure(rho, sys::OpenSystem, ls::LindbladOperator) = map(op -> measure(rho, op, ls), transformed_measurements(sys))
measure(rho, op, ls::LindbladOperator) = map(d -> measure_dissipator(rho, op, d, ls), ls.dissipators)

function measure_dissipator(rho, op, dissipator, system)
    map(superop -> measure(rho, op, superop, system), dissipator)
end
measure(rho, op, dissipator, ls::LindbladOperator) = dot(op, tomatrix(dissipator.du * internal_rep(rho, ls), ls))



lindblad_with_normalizer_dense(lindblad::AbstractMatrix, kv::AbstractVectorizer) = vcat(lindblad, transpose(kv.idvec))

_lindblad_with_normalizer(lindblad, kv::AbstractVectorizer) = (out, in) -> _lindblad_with_normalizer(out, in, lindblad, kv)
_lindblad_with_normalizer_adj(lindblad, kv::AbstractVectorizer) = (out, in) -> _lindblad_with_normalizer_adj(out, in, lindblad, kv)

_lindblad_with_normalizer(out, in, lindblad, kv::KronVectorizer) = (mul!((@view out[1:end-1]), lindblad, in); out[end] = trnorm(in, kv.size); return out)
_lindblad_with_normalizer_adj(out, in, lindblad, kv::KronVectorizer) = (mul!(out, lindblad', (@view in[1:end-1])); out .+= in[end] * kv.idvec; return out)
_lindblad_with_normalizer(out, in, lindblad, krv::KhatriRaoVectorizer) = (mul!((@view out[1:end-1]), lindblad, in); out[end] = dot(krv.idvec, in); return out)
_lindblad_with_normalizer_adj(out, in, lindblad, krv::KhatriRaoVectorizer) = (mul!(out, lindblad', (@view in[1:end-1])); out .+= in[end] .* krv.idvec; return out)

function lindblad_with_normalizer(lindblad, vectorizer)
    newmult! = QuantumDots._lindblad_with_normalizer(lindblad, vectorizer)
    newmultadj! = QuantumDots._lindblad_with_normalizer_adj(lindblad, vectorizer)
    n = size(lindblad, 2)
    lm! = QuantumDots.LinearMap{ComplexF64}(newmult!, newmultadj!, n + 1, n; ismutating=true)
    return Matrix(lm!)
end
lindblad_with_normalizer(lindblad::AbstractMatrix, vectorizer) = lindblad_with_normalizer_dense(lindblad, vectorizer)

identity_density_matrix(system::LindbladOperator) = one(eltype(system.total.du)) * (system.vectorizer.idvec ./ sqrt(size(system.total.du, 2)))