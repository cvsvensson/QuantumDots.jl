struct Lindblad <: AbstractOpenSolver end


struct LindbladSystem{T,U,DS,V,H,C} <: AbstractOpenSystem
    total::T
    unitary::U
    dissipators::DS
    vectorizer::V
    hamiltonian::H
    cache::C
end
struct LindbladCache{KC,MC,SC,OC}
    kroncache::KC
    mulcache::MC
    superopcache::SC
    opcache::OC
end
function LindbladSystem(hamiltonian, leads, vectorizer=default_vectorizer(hamiltonian); rates=map(l -> 1, leads))
    diagham = diagonalize(hamiltonian)
    # diageigvals = Diagonal(eigenvalues(system))
    commutator_hamiltonian = commutator(hamiltonian, vectorizer)
    unitary = -1im * commutator_hamiltonian
    energies = eigenvalues(diagham)
    kroncache = Matrix(unitary)
    superopcache = deepcopy(kroncache)
    mulcache = (complex(Matrix(hamiltonian)))
    opcache = (complex(Matrix(hamiltonian)))
    cache = LindbladCache(kroncache, mulcache, superopcache, opcache)
    dissipators = map((lead, rate) -> LindbladDissipator(superoperator(lead, diagham, rate, vectorizer, cache), rate, lead, diagham, vectorizer, cache), leads, rates)
    total = lindblad_matrix(unitary, dissipators)
    LindbladSystem(total, unitary, dissipators, vectorizer, hamiltonian, cache)
end

struct LindbladDissipator{S,T,L,H,V,C} <: AbstractDissipator
    superop::S
    rate::T
    lead::L
    ham::H
    vectorizer::V
    cache::C
end
Base.adjoint(d::LindbladDissipator) = LindbladDissipator(adjoint(d.superop), adjoint(d.rate), adjoint(d.lead), adjoint(d.ham), d.vectorizer, d.cache)
_dissipator_params(d::LindbladDissipator) = (; μ=d.lead.μ, T=d.lead.T, rate=d.rate)
_dissipator_params(d::LindbladDissipator, p) = (; μ=get(p, :μ, d.lead.μ), T=get(p, :T, d.lead.T), rate=get(p, :rate, d.rate))

function superoperator(lead, diagham::DiagonalizedHamiltonian, rate, vectorizer, cache::LindbladCache)
    superop = zero(cache.superopcache)
    for op in lead.jump_in
        superop .+= superoperator!(op, diagham, lead.T, lead.μ, rate, vectorizer, cache)
        # superop .+= superoperator(op, diagham, lead.T, lead.μ, rate, vectorizer)
    end
    for op in lead.jump_out
        superop .+= superoperator!(op, diagham, lead.T, -lead.μ, rate, vectorizer, cache)
        # superop .+= superoperator(op, diagham, lead.T, -lead.μ, rate, vectorizer)
    end
    return superop
end
function superoperator(lead, diagham, rate, vectorizer, tmp::Nothing)
    sum(superoperator(op, diagham, lead.T, lead.μ, rate, vectorizer) for op in lead.jump_in) .+ sum(superoperator(op, diagham, lead.T, -lead.μ, rate, vectorizer) for op in lead.jump_out)
end
function superoperator(lead_op, diagham, T, μ, rate, vectorizer)
    op = ratetransform(lead_op, diagham, T, μ)
    return dissipator(op, rate, vectorizer)
end

function superoperator!(lead_op, diagham, T, μ, rate, vectorizer, cache::LindbladCache)
    ratetransform!(cache.opcache, lead_op, diagham, T, μ)
    return dissipator!(cache.superopcache, cache.opcache, rate, vectorizer, cache.kroncache, cache.mulcache)
end

function update_coefficients(d::LindbladDissipator, p, tmp=d.cache)
    rate = get(p, :rate, d.rate)
    newlead = update_lead(d.lead, p)
    newsuperop = superoperator(newlead, d.ham, rate, d.vectorizer, tmp)
    LindbladDissipator(newsuperop, rate, newlead, d.ham, d.vectorizer, d.cache)
end

function lindblad_matrix(unitary, dissipators)
    total = zeros(promote(eltype(unitary), map(eltype, dissipators)...)[1], size(unitary)...)
    total .+= (unitary)
    for d in dissipators
        total .+= d.superop
    end
    return total
end

update_lindblad_system(L::LindbladSystem, ::SciMLBase.NullParameters) = L
function update_lindblad_system(L::LindbladSystem, p, tmp=L.cache)
    _newdissipators = map(lp -> first(lp) => update_coefficients(L.dissipators[first(lp)], last(lp), tmp), collect(pairs(p)))
    newdissipators = merge(L.dissipators, _newdissipators)
    total = lindblad_matrix(L.unitary, newdissipators)
    LindbladSystem(total, L.unitary, newdissipators, L.vectorizer, L.hamiltonian, L.cache)
end

LinearOperator(L::LindbladSystem, p=SciMLBase.NullParameters(); normalizer=false) = MatrixOperator(L, p; normalizer)

function MatrixOperator(L::LindbladSystem, p=SciMLBase.NullParameters(); normalizer, kwargs...)
    A0 = Matrix(update_coefficients(L, p))
    A = normalizer ? lindblad_with_normalizer(A0, L.vectorizer) : A0
    MatrixOperator(A)
end

(L::LindbladSystem)(u, p, t; kwargs...) = update_lindblad_system(L, p; kwargs...) * u
update_coefficients(L::LindbladSystem, p) = update_lindblad_system(L, p)

tomatrix(rho::AbstractVector, system::LindbladSystem) = tomatrix(rho, system.vectorizer)
tomatrix(rho::AbstractVector, vectorizer::KronVectorizer) = reshape(rho, vectorizer.size, vectorizer.size)
tomatrix(rho::AbstractVector, vectorizer::KhatriRaoVectorizer) = BlockDiagonal(map((size, inds) -> reshape(rho[inds], size, size), vectorizer.sizes, vectorizer.vectorinds))
internal_rep(rho, system::LindbladSystem) = internal_rep(rho, system.vectorizer)
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

# const DENSE_CUTOFF = 16
# const KR_LAZY_CUTOFF = 40
dissipator(L, krv::KhatriRaoVectorizer) = khatri_rao_dissipator(L, krv)
commutator(A, krv::KhatriRaoVectorizer) = khatri_rao_commutator(A, krv)

function dissipator(L, kv::KronVectorizer)
    D = (conj(L) ⊗ L - 1 / 2 * kronsum(transpose(L' * L), L' * L))
    return Matrix(D)
end

dissipator!(out, L::AbstractMatrix, rate, kv::KhatriRaoVectorizer, kroncache, mulcache) = khatri_rao_dissipator!(out, L, rate, kv, kroncache, mulcache)
dissipator(L::AbstractMatrix, rate, kv::KhatriRaoVectorizer) = khatri_rao_dissipator(L, kv; rate)

function dissipator!(out, L::AbstractMatrix{T}, rate, kv::KronVectorizer, kroncache, mulcache) where {T}
    kron!(kroncache, transpose(L'), L)
    out .= kroncache
    i = I(kv.size)
    mul!(mulcache, L', L, 1 / 2, 0)
    kron!(kroncache, transpose(mulcache), i)
    out .-= kroncache
    kron!(kroncache, i, mulcache)
    out .-= kroncache
    out .*= rate
    return out
end
function dissipator(L, rate, kv::KronVectorizer)
    return Matrix(rate * (conj(L) ⊗ L - 1 / 2 * kronsum(transpose(L' * L), L' * L)))
end

commutator(A, ::KronVectorizer) = commutator(A)
commutator(A) = kron(one(A), A) - kron(transpose(A), one(A))
measure(rho, op, ls::AbstractOpenSystem) = map(d -> measure(rho, op, d, ls), ls.dissipators)

Base.Matrix(d::LindbladDissipator) = d.superop
Base.Matrix(L::LindbladSystem) = L.total

measure(rho, op, dissipator::AbstractDissipator, ls::AbstractOpenSystem) = dot(op, tomatrix(dissipator * internal_rep(rho, ls), ls))

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

identity_density_matrix(system::LindbladSystem) = one(eltype(system.total)) * (system.vectorizer.idvec ./ sqrt(size(system.total, 2)))


function ODEProblem(system::LindbladSystem, u0, tspan, p=SciMLBase.NullParameters(), args...; kwargs...)
    internalu0 = internal_rep(u0, system)
    ODEProblem(LinearOperator(system, p; kwargs...), internalu0, tspan, p, args...; kwargs...)
end

Base.size(ls::LindbladSystem) = size(ls.total)