struct Lindblad <: AbstractOpenSolver end


struct LindbladSystem{T,U,DS,V,H,C} <: AbstractOpenSystem
    total::T
    unitary::U
    dissipators::DS
    vectorizer::V
    hamiltonian::H
    cache::C
end
struct LindbladCache{KC, MC, SC, OC} 
    kroncache::KC
    mulcache::MC
    superopcache::SC
    opcache::OC
end
function LindbladSystem(system::OpenSystem{<:DiagonalizedHamiltonian}, vectorizer=default_vectorizer(system); rates = map(l-> one(eltype(system)), system.leads))
    commutator_hamiltonian = commutator(eigenvalues(system), vectorizer)
    unitary = -1im * commutator_hamiltonian
    energies = eigenvaluevector(system)
    kroncache = DiffCache(Matrix(unitary))
    superopcache = deepcopy(kroncache)
    mulcache = DiffCache(complex(Matrix(eigenvalues(system))))
    opcache = DiffCache(complex(Matrix(eigenvalues(system))))
    cache = LindbladCache(kroncache, mulcache, superopcache, opcache)
    _cache = get_cache(cache, map((l,rate) -> (l.μ, l.T, rate), system.leads,rates))
    dissipators = map((lead,rate) -> dissipator(lead, energies, rate,vectorizer,  cache), system.leads,rates)
    total = deepcopy(_cache.superopcache)
    lindblad_matrix!(total, unitary, dissipators)
    LindbladSystem(total, unitary, dissipators, vectorizer, system.hamiltonian, cache)
end

struct LindbladDissipator{S,T,L,E,V,C} <: AbstractDissipator
    superop::S
    rate::T
    lead::L
    energies::E
    vectorizer::V
    cache::C
end

using ForwardDiff
_dissipator_params(d::LindbladDissipator) = (;μ = d.lead.μ, T = d.lead.T, rate = d.rate)
_dissipator_params(d::LindbladDissipator,p) = (;μ = get(p,:μ, d.lead.μ), T = get(p,:T,d.lead.T), rate = get(p,:rate,d.rate))
function chem_derivative(d)
    func = μ -> Matrix(update(d,(;μ)))
    ForwardDiff.derivative(func, d.lead.μ)
end
function chem_derivative(d, _p)
    p = _dissipator_params(d,_p)
    func = μ -> Matrix(update(d,(;μ,T = p.T, rate=p.rate)))
    ForwardDiff.derivative(func, p.μ)
end

function update_lead(lead, props)
    μ = get(props, :μ, lead.μ)
    T = get(props, :T, lead.T)  
    NormalLead(lead; μ, T)
end
function dissipator(lead, energies, rate,vectorizer, _cache::LindbladCache)
    cache = get_cache(_cache, (lead.T,lead.μ,rate))
    ratetransform!(cache.opcache, lead.jump_in, energies, lead.T, lead.μ)
    superop_in = deepcopy(dissipator!(cache.superopcache, cache.opcache, rate, vectorizer, cache.kroncache, cache.mulcache))
    ratetransform!(cache.opcache, lead.jump_out, energies, lead.T, -lead.μ)
    superop_out = deepcopy(dissipator!(cache.superopcache, cache.opcache, rate, vectorizer, cache.kroncache, cache.mulcache))
    superop = (;in = superop_in, out = superop_out, total = superop_out + superop_in)
    LindbladDissipator(superop, rate,lead, energies, vectorizer, _cache)
end


function get_cache(L::LindbladCache, u)
    t = promote(Iterators.flatten(u)...)[1]
    LindbladCache(map(field->get_tmp(getproperty(L, field), t), fieldnames(LindbladCache))...)
end
function update(d::LindbladDissipator, p)
    rate = get(p, :rate, d.rate)
    newlead = update_lead(d.lead, p)
    return dissipator(newlead, d.energies, rate, d.vectorizer, d.cache)
end


function lindblad_matrix!(total,unitary,dissipators) 
    total .= unitary
    for d in dissipators
        total .+= d.superop.total
    end
    return total
end

update_lindblad_system(L::LindbladSystem, ::SciMLBase.NullParameters) = L
function update_lindblad_system(L::LindbladSystem, p)
    _newdissipators = map(lp-> first(lp) => update(L.dissipators[first(lp)], last(lp)), collect(pairs(p)))
    newdissipators = merge(L.dissipators, _newdissipators)
    T = promote_type(eltype(L.unitary), map(d->eltype(d.superop.total), newdissipators)...)
    total = zeros(T,size(L.cache.superopcache.du)...)
    lindblad_matrix!(total,L.unitary,newdissipators)
    LindbladSystem(total,L.unitary,newdissipators, L.vectorizer, L.hamiltonian, L.cache)
end 

LindbladOperator(sys::OpenSystem, vectorizer=default_vectorizer(sys)) = LindbladSystem(sys, vectorizer)
LinearOperator(L::LindbladSystem, p=SciMLBase.NullParameters(); normalizer=false) = MatrixOperator(L, p; normalizer)
function MatrixOperator(L::LindbladSystem, p::SciMLBase.NullParameters; normalizer)
    A = normalizer ? lindblad_with_normalizer(L.total, L.vectorizer) : L.total
    MatrixOperator(A)
end
function MatrixOperator(L::LindbladSystem, p=SciMLBase.NullParameters(); normalizer)
    A0 = L(p).total
    A = normalizer ? lindblad_with_normalizer(A0, L.vectorizer) : A0
    MatrixOperator(A)
end

(L::LindbladSystem)(u, p, t; kwargs...) = update_lindblad_system(L, p; kwargs...) * u
update(L::LindbladSystem,p) = update_lindblad_system(L,p)

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

const DENSE_CUTOFF = 16
const KR_LAZY_CUTOFF = 40
dissipator(L, krv::KhatriRaoVectorizer) = sum(krv.sizes) > KR_LAZY_CUTOFF ? khatri_rao_lazy_dissipator(L, krv.sizes) : khatri_rao_dissipator(L, krv.sizes)
commutator(A, krv::KhatriRaoVectorizer) = sum(krv.sizes) > KR_LAZY_CUTOFF ? khatri_rao_lazy_commutator(A, krv.sizes) : khatri_rao_commutator(A, krv.sizes)

function dissipator(L, kv::KronVectorizer)
    D = (conj(L) ⊗ L - 1 / 2 * kronsum(transpose(L' * L), L' * L))
    return kv.size > DENSE_CUTOFF ? D : Matrix(D)
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
measure(rho, sys::OpenSystem, ls::AbstractOpenSystem) = map(op -> measure(rho, op, ls), transformed_measurements(sys))
measure(rho, op, ls::AbstractOpenSystem) = map(d -> measure_dissipator(rho, op, d, ls), ls.dissipators)

function measure_dissipator(rho, op, dissipator, system)
    measure(rho, op, Matrix(dissipator), system)
end
Base.Matrix(d::LindbladDissipator) = d.superop.total
Base.Matrix(L::LindbladSystem) = L.total

measure(rho, op, dissipator, ls::AbstractOpenSystem) = dot(op, tomatrix(dissipator * internal_rep(rho, ls), ls))

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
