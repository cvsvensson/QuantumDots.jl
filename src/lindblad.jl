struct Lindblad <: AbstractOpenSolver end
# struct LindbladSystem{O,U,Ds,L,V} <: AbstractOpenSystem
#     system::O
#     unitary::U
#     dissipators::Ds
#     lindblad::L
#     vectorizer::V
# end
# leads(ls::LindbladSystem) = leads(ls.system)
# measurements(ls::LindbladSystem) = measurements(ls.system)
# transformed_measurements(ls::LindbladSystem) = transformed_measurements(ls.system)

struct FermionDissipator{L,M,D,V,T,E,H}
    lead::L
    opin::M
    opout::M
    superopin::D
    superopout::D
    vectorizer::V
    props::LArray{T,1, Vector{T}, (:T, :μ, :rate)}
    energies::E
    commutator_hamiltonian::H
    kroncache::D
    mulcache::M
    function FermionDissipator(lead::L, vectorizer::V, energies::E) where {L,V,E}
        props = _default_lindblad_dissipator_params(lead)
        commutator_hamiltonian = commutator(Diagonal(energies), vectorizer)
        opin = ratetransform(lead.jump_in, energies, props.T, props.μ)
        opout = ratetransform(lead.jump_out, energies, props.T, -props.μ)
        M = typeof(opin)
        superopin = dissipator(opin, vectorizer)
        superopout = dissipator(opout, vectorizer)
        kroncache = deepcopy(superopin)
        mulcache = deepcopy(opin)
        H = typeof(commutator_hamiltonian)
        new{L,M,typeof(superopin),V,eltype(props),E,H}(lead, opin, opout, superopin, superopout, vectorizer, props, energies, commutator_hamiltonian, kroncache, mulcache)
    end
end

chemical_potential(d::FermionDissipator) = d.props.μ
temperature(d::FermionDissipator) = d.props.T
rate(d::FermionDissipator) = d.props.rate
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

function update_dissipator!(d::FermionDissipator, p = ())
    if haskey(p, :μ) || haskey(p, :T)
        μ = get(p, :μ, d.props.μ)
        T = get(p, :T, d.props.T)
        d.props.μ = μ
        d.props.T = T
        ratetransform!(d.opin, d.lead.jump_in, d.energies, T, μ) 
        ratetransform!(d.opout, d.lead.jump_out, d.energies, T, -μ)
    end 
    dissipator!(d.superopin, d.opin, d.vectorizer, d.kroncache, d.mulcache)
    dissipator!(d.superopout, d.opout, d.vectorizer, d.kroncache, d.mulcache)
    if haskey(p,:rate)
        d.props.rate = p.rate     
    end
    rate = d.props.rate
    d.superopin .*= rate
    d.superopout .*= rate
    return nothing
end


struct LindbladOperator{T,V,D,U,H} <: AbstractOpenSystem
    hamiltonian::H
    unitary::U
    dissipators::D
    total::T
    vectorizer::V
end
function LindbladOperator(hamiltonian::DiagonalizedHamiltonian, leads, vectorizer = default_vectorizer(hamiltonian))
    commutator_hamiltonian = commutator(eigenvalues(hamiltonian), vectorizer)
    unitary = -1im * commutator_hamiltonian
    dissipators = map(lead -> FermionDissipator(lead, vectorizer, diag(hamiltonian.eigenvalues)), leads)
    total = similar(unitary, size(unitary)) #Probably only works for dense matrices now
    total .= unitary
    for d in dissipators
        total .+= d.superopin .+ d.superopout
    end
    LindbladOperator(hamiltonian, unitary, dissipators, total, vectorizer)
end
LindbladOperator(sys::OpenSystem, vectorizer = default_vectorizer(sys)) = LindbladOperator(sys.hamiltonian, sys.leads, vectorizer)
update_dissipator!(L, label, props) = update_dissipator!(L.dissipators[label], props)
LinearAlgebra.mul!(out, L::LindbladOperator, in) = mul!(out, L.total, in)

# MatrixOperator(::Lindblad, sys::OpenSystem)
# LinearOperatorWithOutNormalizer(L::LindbladOperator{<:AbstractMatrix}) = MatrixOperator(L)
# LinearOperatorWithNormalizer(L::LindbladOperator{<:AbstractMatrix}) = MatrixOperatorWithNormalizer(L)
LinearOperator(L::LindbladOperator{<:AbstractMatrix}; normalizer = false) = MatrixOperator(L; normalizer)
function MatrixOperator(_L::LindbladOperator; normalizer)
    L = deepcopy(_L)
    update_func! = lindblad_updater!(L; normalizer)
    A = normalizer ? lindblad_with_normalizer(L.total, L.vectorizer) : L.total
    MatrixOperator(A; update_func!)
end

_pairs(p) = pairs(p)
_pairs(::SciMLBase.NullParameters) = pairs(())

function lindblad_updater!(L; normalizer)
    normalizer || return function update_func!(A, u, p, t)
        updated = false
        for (label, props) in _pairs(p)
            update_dissipator!(L, label, props)
            updated = true
        end
        if updated
            update_total_operator!(L)
            A .= L.total
        end
        return nothing
    end

    return function update_func_normalizer!(A, u, p, t)
        updated = false
        for (label, props) in _pairs(p)
            update_dissipator!(L, label, props)
            updated = true
        end
        if updated
            update_total_operator!(L)
            A[1:end-1,:] .= L.total
        end
        return nothing
    end
end

function update_total_operator!(L::LindbladOperator)
    L.total .= L.unitary
    for d in L.dissipators
        L.total .+= d.superopin .+ d.superopout
    end
end
# function LinearProblem(::Lindblad, system::OpenSystem; kwargs...)
#     ls = prepare_lindblad(system; kwargs...)
#     LinearProblem(ls; kwargs...)
# end


tomatrix(rho::AbstractVector, system::LindbladOperator) = tomatrix(rho, system.vectorizer)
tomatrix(rho::AbstractVector, vectorizer::KronVectorizer) = reshape(rho, vectorizer.size, vectorizer.size)
tomatrix(rho::AbstractVector, vectorizer::KhatriRaoVectorizer) = BlockDiagonal(map((size, inds) -> reshape(rho[inds], size, size), vectorizer.sizes, vectorizer.vectorinds))
internal_rep(rho, system::LindbladOperator) = internal_rep(rho, system.vectorizer)
internal_rep(rho::AbstractMatrix, ::KronVectorizer) = vec(rho)
internal_rep(rho::UniformScaling, v::KronVectorizer) = vec(Diagonal(rho,v.size))
internal_rep(rho::UniformScaling, v::KhatriRaoVectorizer) = vecdp(BlockDiagonal([Diagonal(rho, sz) for sz in  v.sizes]))
internal_rep(rho::BlockDiagonal, vectorizer::KhatriRaoVectorizer) = iscompatible(rho,vectorizer) ? vecdp(rho) : internal_rep(Matrix(rho), vectorizer)
iscompatible(rho::BlockDiagonal, vectorizer::KhatriRaoVectorizer) = size.(rho.blocks, 1) == size.(rho.blocks, 2) == vectorizer.sizes 
function internal_rep(rho::AbstractMatrix{T}, vectorizer::KhatriRaoVectorizer) where T
    inds = vectorizer.inds 
    indsv = vectorizer.vectorinds
    v = Vector{T}(undef, vectorizer.cumsumsquared[end])
    for i in eachindex(inds)
        v[indsv[i]] = vec(rho[inds[i],inds[i]])
    end
    return v
end
internal_rep(rho::AbstractVector, ::AbstractVectorizer) = rho

# LinearOperator(system::LindbladSystem; kwargs...) = LinearOperator(system.lindblad; kwargs...)
# LinearOperatorWithNormalizer(system::LindbladSystem; kwargs...) = LinearOperator(lindblad_with_normalizer(system.lindblad, system.vectorizer); kwargs...)

# Base.show(io::IO, ::MIME"text/plain", system::LindbladSystem) = show(io, system)
# Base.show(io::IO, system::LindbladSystem) = print(io, "LindbladSystem:", "\nOpenSystem", repr(system.system),
#     "\nUnitary: ", reprlindblad(system.unitary), "\ndissipators: ", reprdissipators(system.dissipators),
#     "\nlindblad: ", reprlindblad(system.lindblad),
#     "\nvectorizer: ", typeof(system.vectorizer))

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
function dissipator!(out, L::AbstractMatrix{T}, kv::KronVectorizer, kroncache, mulcache) where T
    kron!(kroncache, conj(L), L)
    out .= kroncache
    i = Diagonal{T}(I, kv.size)
    mul!(mulcache, L',L,1/2*one(T),zero(T))
    kron!(kroncache, mulcache, i)
    out .-= kroncache
    kron!(kroncache, i, mulcache)
    out .-= kroncache
    #D = (conj(L) ⊗ L - 1 / 2 * kronsum(transpose(L' * L), L' * L))
    #return kv.size > DENSE_CUTOFF ? D : Matrix(D)
    return out
end
commutator(A, ::KronVectorizer) = commutator(A)
commutator(A) = kron(one(A), A) - kron(transpose(A), one(A))
measure(rho, sys::OpenSystem, ls::LindbladOperator) = map(op-> measure(rho, op, ls), transformed_measurements(sys))
measure(rho, op, ls::LindbladOperator) = map(d -> measure_dissipator(rho, op, d,ls), ls.dissipators)

# function measure_dissipator(rho, op, dissipator::NamedTuple{(:in, :out, :label),<:Any}, system)
#     results = map(dissipator_op -> measure(rho, op, dissipator_op, system), (; dissipator.in, dissipator.out))
#     merge(results, (; total=sum(results), label=dissipator.label))
# end
function measure_dissipator(rho, op, dissipator, system)
    map(dissipator_op -> measure(rho, op, dissipator_op, system), (;in = dissipator.superopin, out = dissipator.superopout))
end
measure(rho, op, dissipator, ls::LindbladOperator) = dot(op, tomatrix(dissipator * internal_rep(rho, ls), ls))
# current(ρ,op,sj) = tr(op * reshape(sj*ρ,size(op)))
# commutator(T1,T2) = -T1⊗T2 + T2⊗T1
# commutator(A) = -transpose(A)⊗one(A) + one(A)⊗A


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

identity_density_matrix(system::LindbladOperator) = one(eltype(system.total)) * (system.vectorizer.idvec ./ sqrt(size(system.total, 2)))

# function prepare_lindblad(system::OpenSystem; kwargs...)
#     diagonalsystem = diagonalize(system; kwargs...)
#     prepare_lindblad(diagonalsystem; kwargs...)
# end
# function prepare_lindblad(diagonalsystem::OpenSystem{<:DiagonalizedHamiltonian}; kwargs...)
#     transformedsystem = ratetransform(diagonalsystem)
#     vectorizer = default_vectorizer(hamiltonian(diagonalsystem))
#     dissipators = map(lead -> dissipator_from_transformed_lead(lead, vectorizer), transformed_leads(transformedsystem))
#     unitary = -1im * commutator(eigenvalues(transformedsystem), vectorizer)
#     lindblad = unitary + sum(d -> d.in + d.out, dissipators)
#     lindbladsystem = LindbladSystem(transformedsystem, unitary, dissipators, lindblad, vectorizer)
#     return lindbladsystem
# end

# function dissipator_from_transformed_lead(lead::NormalLead, vectorizer::AbstractVectorizer, props)
#     opin = dissipator(lead.jump_in, vectorizer)
#     opout = dissipator(lead.jump_out, vectorizer)
#     (; in=opin, out=opout, label=lead.label)
# end
