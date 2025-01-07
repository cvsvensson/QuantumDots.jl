struct Lindblad <: AbstractOpenSolver end


"""
    struct LindbladSystem{T,U,DS,V,H,C} <: AbstractOpenSystem

A struct representing a Lindblad open quantum system.

# Fields
- `total::T`: The total lindblad matrix operator.
- `unitary::U`: The unitary part of the system.
- `dissipators::DS`: The dissipators of the system.
- `vectorizer::V`: The vectorizer used for the system.
- `hamiltonian::H`: The Hamiltonian of the system.
- `cache::C`: The cache used for the system.
"""
struct LindbladSystem{T,U,DS,V,H,MH,C} <: AbstractOpenSystem
    total::T
    unitary::U
    dissipators::DS
    vectorizer::V
    hamiltonian::H
    matrixhamiltonian::MH
    cache::C
end
"""
    struct LindbladCache{KC, MC, SC, OC}

A cache structure used in the Lindblad equation solver.

# Fields
- `kroncache::KC`: Cache for Kronecker products.
- `mulcache::MC`: Cache for matrix multiplications.
- `superopcache::SC`: Cache for superoperators.
- `opcache::OC`: Cache for operators.

"""
struct LindbladCache{KC,MC,SC,MC1,MC2,BC}
    kroncache::KC
    mulcache::MC
    superopcache::SC
    matrixcache1::MC1
    matrixcache2::MC2
    blockcache::BC
end
# """
#     LindbladCache(superoperator, operator)

# Constructs a cache for the LindbladSystem.
# """
# function LindbladCache(superoperator, operator)
#     kroncache = Matrix(superoperator)
#     superopcache = deepcopy(kroncache)
#     mulcache = (complex(Matrix(operator))) # 
#     opcache = deepcopy(mulcache)
#     LindbladCache(kroncache, mulcache, superopcache, opcache, deepcopy(opcache))
# end
"""
    LindbladSystem(hamiltonian, leads, vectorizer=default_vectorizer(hamiltonian); rates=map(l -> 1, leads), usecache=false)

Constructs a Lindblad system for simulating open quantum systems.

## Arguments
- `hamiltonian`: The Hamiltonian of the system.
- `leads`: An list of operators representing the leads.
- `vectorizer`: Determines how to vectorize the lindblad equation. Defaults to `default_vectorizer(hamiltonian)`.
- `rates`: An array of rates for each lead. Defaults to an array of ones with the same length as `leads`.
- `usecache`: A boolean indicating whether to use a cache. Defaults to `false`.
"""
function LindbladSystem(hamiltonian, leads, vectorizer=default_vectorizer(hamiltonian); rates=map(l -> 1, leads), usecache=false)
    diagham = diagonalize(hamiltonian)
    matrixdiagham = DiagonalizedHamiltonian(collect(diagham.values), collect(diagham.vectors), collect(hamiltonian))
    commutator_hamiltonian = Matrix(commutator(hamiltonian, vectorizer))
    unitary = -1im * commutator_hamiltonian

    matrixcache1 = similar(matrixdiagham.original)
    matrixcache2 = similar(matrixcache1)
    blockcache = similar(diagham.vectors)
    superopcache = similar(unitary)
    mulcache = similar(matrixcache1)
    kroncache = materialize_kroncache(eltype(unitary), vectorizer)
    cache = LindbladCache(kroncache, mulcache, superopcache, matrixcache1, matrixcache2, blockcache)
    cache = usecache ? cache : nothing

    dissipators = map((lead, rate) -> LindbladDissipator(superoperator(lead, diagham, rate, vectorizer, cache), rate, lead, diagham, matrixdiagham, vectorizer, cache), leads, rates)
    total = lindblad_matrix(unitary, dissipators)
    LindbladSystem(total, unitary, dissipators, vectorizer, hamiltonian, matrixdiagham, cache)
end
materialize_kroncache(T, kv::KronVectorizer) = zeros(T, kv.size^2, kv.size^2)

"""
    struct LindbladDissipator{S,T,L,H,V,C} <: AbstractDissipator

A struct representing a Lindblad dissipator.

# Fields
- `superop::S`: The superoperator representing the dissipator.
- `rate::T`: The rate of the dissipator.
- `lead::L`: The lead associated with the dissipator.
- `ham::H`: The Hamiltonian associated with the dissipator.
- `vectorizer::V`: The vectorizer used for vectorization.
- `cache::C`: The cache used for storing intermediate results.
"""
struct LindbladDissipator{S,T,L,H,MH,V,C} <: AbstractDissipator
    superop::S
    rate::T
    lead::L
    ham::H
    matrixham::MH
    vectorizer::V
    cache::C
end
Base.adjoint(d::LindbladDissipator) = LindbladDissipator(adjoint(d.superop), adjoint(d.rate), adjoint(d.lead), adjoint(d.ham), adjoint(d.matrixham), d.vectorizer, d.cache)
_dissipator_params(d::LindbladDissipator) = (; μ=d.lead.μ, T=d.lead.T, rate=d.rate)
_dissipator_params(d::LindbladDissipator, p) = (; μ=get(p, :μ, d.lead.μ), T=get(p, :T, d.lead.T), rate=get(p, :rate, d.rate))

function superoperator(lead, diagham::DiagonalizedHamiltonian, rate, vectorizer, cache::LindbladCache)
    superop = zero(cache.superopcache)
    superoperator!(superop, lead, diagham, rate, vectorizer, cache)
end
struct ADD_SUPEROP end
struct RESET_SUPEROP end
function superoperator!(superop, lead, diagham::DiagonalizedHamiltonian, rate, vectorizer, cache::LindbladCache, mode=RESET_SUPEROP())
    if mode == RESET_SUPEROP()
        fill!(superop, zero(eltype(superop)))
    end
    for op in lead.jump_in
        superoperator!(superop, op, diagham, lead.T, lead.μ, rate, vectorizer, cache, ADD_SUPEROP())
        # superop .+= newop
        #TODO: make a version of superoperator! that adds the new superoperator to the old, so we can avoid copying the result again
    end
    for op in lead.jump_out
        superoperator!(superop, op, diagham, lead.T, -lead.μ, rate, vectorizer, cache, ADD_SUPEROP())
    end
    return superop
end
function superoperator(lead, diagham, rate, vectorizer, ::Nothing)
    (sum(superoperator(op, diagham, lead.T, lead.μ, rate, vectorizer) for op in lead.jump_in) + sum(superoperator(op, diagham, lead.T, -lead.μ, rate, vectorizer) for op in lead.jump_out))
end
"""
    superoperator(lead_op, diagham, T, μ, rate, vectorizer)

Construct the superoperator associated with the operator `lead_op`. Transforms the operator to the energy basis and includes fermi-Dirac statistics.

# Arguments
- `lead_op`: The operator representing the lead coupling.
- `diagham`: The diagonal Hamiltonian.
- `T`: The temperature.
- `μ`: The chemical potential.
- `rate`: The rate of the dissipative process.
- `vectorizer`: The vectorizer struct.
"""
function superoperator(lead_op, diagham, T, μ, rate, vectorizer)
    op = ratetransform(lead_op, diagham, T, μ)
    return dissipator(op, rate, vectorizer)
end

function superoperator!(superop, lead_op, diagham, T, μ, rate, vectorizer, cache::LindbladCache, mode)
    op = ratetransform!(cache.matrixcache1, cache.matrixcache2, lead_op, diagham, T, μ)
    return dissipator!(superop, op, rate, vectorizer, cache.superopcache, cache.kroncache, cache.mulcache, mode)
end

function update_coefficients(d::LindbladDissipator, p, tmp=d.cache)
    rate = get(p, :rate, d.rate)
    newlead = update_lead(d.lead, p)
    newsuperop = superoperator(newlead, d.matrixham, rate, d.vectorizer, tmp)
    LindbladDissipator(newsuperop, rate, newlead, d.ham, d.matrixham, d.vectorizer, d.cache)
end
function update_coefficients!(d::LindbladDissipator, p, cache=d.cache)
    rate = get(p, :rate, d.rate)
    newlead = update_lead(d.lead, p)
    newsuperop = superoperator!(d.superop, newlead, d.matrixham, rate, d.vectorizer, cache)
    LindbladDissipator(newsuperop, rate, newlead, d.ham, d.matrixham, d.vectorizer, d.cache)
end

function lindblad_matrix(unitary, dissipators)
    total = zeros(promote(eltype(unitary), map(eltype, dissipators)...)[1], size(unitary)...)
    total .+= (unitary)
    for d in dissipators
        total .+= d.superop
    end
    return total
end
function lindblad_matrix!(total, unitary, dissipators)
    fill!(total, zero(eltype(total)))
    total .+= unitary
    for d in dissipators
        total .+= d.superop
    end
    return total
end

update_coefficients(L::LindbladSystem, ::SciMLBase.NullParameters) = L
function update_coefficients(L::LindbladSystem, p, tmp=L.cache)
    _newdissipators = map(lp -> first(lp) => update_coefficients(L.dissipators[first(lp)], last(lp), tmp), collect(pairs(p)))
    newdissipators = merge(L.dissipators, _newdissipators)
    total = lindblad_matrix(L.unitary, newdissipators)
    LindbladSystem(total, L.unitary, newdissipators, L.vectorizer, L.hamiltonian, L.matrixhamiltonian, L.cache)
end
function update_coefficients!(L::LindbladSystem, p, cache=L.cache)
    # println("__")
    _newdissipators = map(lp -> first(lp) => update_coefficients!(L.dissipators[first(lp)], last(lp), cache), collect(pairs(p)))
    newdissipators = merge(L.dissipators, _newdissipators)
    total = lindblad_matrix!(L.total, L.unitary, newdissipators)
    LindbladSystem(total, L.unitary, newdissipators, L.vectorizer, L.hamiltonian, L.matrixhamiltonian, L.cache)
end

LinearOperator(L::LindbladSystem, p=SciMLBase.NullParameters(); normalizer=false) = MatrixOperator(L, p; normalizer)

function MatrixOperator(L::LindbladSystem, p=SciMLBase.NullParameters(); normalizer, kwargs...)
    A0 = Matrix(update_coefficients(L, p))
    A = normalizer ? lindblad_with_normalizer(A0, L.vectorizer) : A0
    MatrixOperator(A)
end

(L::LindbladSystem)(u, p, t; kwargs...) = update_lindblad_system(L, p; kwargs...) * u

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

dissipator!(out, L::AbstractMatrix, rate, kv::KhatriRaoVectorizer, superopcache, kroncache, mulcache, mode) = khatri_rao_dissipator!(out, L, rate, kv, kroncache, mulcache, mode)
dissipator(L::AbstractMatrix, rate, kv::KhatriRaoVectorizer) = khatri_rao_dissipator(L, kv; rate)

dissipator!(superop, L::AbstractMatrix, rate, kv::KronVectorizer, superopcache, kroncache, mulcache, ::RESET_SUPEROP) = dissipator!(superop, L, rate, kv, kroncache, mulcache)
dissipator!(superop, L::AbstractMatrix, rate, kv::KronVectorizer, superopcache, kroncache, mulcache, ::ADD_SUPEROP) = superop .+= dissipator!(superopcache, L, rate, kv, kroncache, mulcache)
function dissipator!(out, L::AbstractMatrix, rate, kv::KronVectorizer, kroncache, mulcache)
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
# function dissipator_linearmap(L, rate, ::KronVectorizer)
#     rate * (conj(L) ⊗ L - 1 / 2 * kronsum(transpose(L' * L), L' * L))
# end
"""
    dissipator(L, rate, kv::KronVectorizer)

Constructs the dissipator associated to the jump operator `L`.
"""
function dissipator(L, rate, kv::KronVectorizer)
    kroncache = kron(transpose(L'), L)
    out = deepcopy(kroncache)
    i = Eye(kv.size)#I(kv.size)
    mulcache = L' * L / 2
    kron!(kroncache, transpose(mulcache), i)
    out .-= kroncache
    kron!(kroncache, i, mulcache)
    out .-= kroncache
    out .*= rate
    return out
end

commutator(A, ::KronVectorizer) = commutator(A)
commutator(A) = kron(one(A), A) - kron(transpose(A), one(A))
measure(rho, op, ls::AbstractOpenSystem) = map(d -> measure(rho, op, d, ls), ls.dissipators)

Base.Matrix(d::LindbladDissipator) = d.superop
Base.Matrix(L::LindbladSystem) = L.total
LinearAlgebra.mul!(v, d::LindbladDissipator, u) = mul!(v, Matrix(d), u)
LinearAlgebra.mul!(v, d::LindbladDissipator, u, a, b) = mul!(v, Matrix(d), u, a, b)
LinearAlgebra.mul!(v, ls::LindbladSystem, u, a, b) = mul!(v, ls.total, u, a, b)
LinearAlgebra.mul!(v, ls::LindbladSystem, u) = mul!(v, ls.total, u)

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
Base.size(ls::LindbladDissipator) = size(ls.superop)
Base.eltype(ls::LindbladSystem) = eltype(ls.total)
Base.eltype(ls::LindbladDissipator) = eltype(ls.superop)