struct LindbladSystem{O,U,Ds,L,V} <: AbstractOpenSystem
    system::O
    unitary::U
    dissipators::Ds
    lindblad::L
    vectorizer::V
end
isdiagonalized(::LindbladSystem{<:DiagonalizedHamiltonian}) = true
isdiagonalized(::LindbladSystem) = false 
leads(ls::LindbladSystem) = leads(ls.system)
transformed_leads(ls::LindbladSystem) = transformed_leads(ls.system)
measurements(ls::LindbladSystem) = measurements(ls.system)
transformed_measurements(ls::LindbladSystem) = transformed_measurements(ls.system)
struct Lindblad <: AbstractOpenSolver end

function LinearProblem(::Lindblad, system::OpenSystem; kwargs...)
    ls = prepare_lindblad(system; kwargs...)
    LinearProblem(ls; kwargs...)
end

struct MatVector{V,T,R} <: AbstractVector{T}
    data::V
    reshaper::R
    function MatVector(data::V, reshaper::R) where {V <: AbstractVector, R}
        new{V,eltype(V),R}(data, reshaper)
    end
end
MatVector{V, T, F}(::UndefInitializer,n) where {V,T,F} = Vector{T}(undef,n)
tomatrix(v::MatVector) = v.reshaper(v.data)
tomatrix(sol::LinearSolution) = tomatrix(sol.u)
Base.size(v::MatVector) = size(v.data)
function Base.getindex(v::MatVector, args...)
    getindex(v.data,args...)
end
function Base.setindex!(v::MatVector,args...)
    setindex!(v.data, args...)
end
Base.similar(m::MatVector) = MatVector(similar(m.data), m.reshaper)
Base.similar(m::MatVector,::Type{S}) where S = MatVector(similar(m.data,S), m.reshaper)
MatVector(m::AbstractVector, system::AbstractOpenSystem) = MatVector(m, Reshaper(system))
Reshaper(system::LindbladSystem) = Reshaper(system.vectorizer)
Reshaper(v::KronVectorizer) = vec->reshape(vec,v.size,v.size)
Reshaper(v::KhatriRaoVectorizer) = vec->BlockDiagonal(map((size, inds) -> reshape(vec[inds], size, size), v.sizes, v.vectorinds))
Reshaper(::PauliSystem) = Diagonal
# function Base.getindex(m::LindbladDensityMatrix{T}, i1::Integer, i2::Integer) where T
#     ind = indexmap(m.vectorizer,i1,i2)
#     isnothing(ind) && return zero(T)
#     m.data[ind]
# end
# function Base.setindex!(m::LindbladDensityMatrix{T}, v, i1::Integer, i2::Integer) where T
#     ind = indexmap(m.vectorizer,i1,i2)
#     isnothing(ind) && return error("Index $((i1,i2)) is not allowed")
#     m.data[ind] = v
# end
# indexmap(v::KronVectorizer,i1,i2) = LinearIndices((v.size,v.size))[i1,i2]
# function indexmap(v::KhatriRaoVectorizer,i1,i2)
#     index = findfirst(ind -> i1 in ind && i2 in ind, v.inds)
#     isnothing(index) && return nothing
#     offset = v.cumsum[index]
#     LinearIndices((v.inds[index],v.inds[index]))[i1-offset, i2-offset] + v.cumsumsquared[index]
# end

# external_rep(rho::AbstractVector, system::LindbladSystem) = external_rep(rho, system.vectorizer)
# external_rep(rho::AbstractVector, vectorizer::KronVectorizer) = reshape(rho, vectorizer.size, vectorizer.size)
# external_rep(rho::AbstractVector, vectorizer::KhatriRaoVectorizer) = BlockDiagonal(map((size, inds) -> reshape(rho[inds], size, size), vectorizer.sizes, sizestoinds(vectorizer.sizes .^ 2)))
internal_rep(rho, system) = MatVector(_internal_vec(rho, system), Reshaper(system))
_internal_vec(rho, system::LindbladSystem) = _internal_vec(rho, system.vectorizer)
_internal_vec(rho, ::KronVectorizer) = vec(rho)
_internal_vec(rho::UniformScaling, v::KronVectorizer) = vec(Diagonal(rho,v.size))
_internal_vec(rho::UniformScaling, v::KhatriRaoVectorizer) = vecdp(BlockDiagonal([Diagonal(rho, sz) for sz in  v.sizes]))
_internal_vec(rho::BlockDiagonal, vectorizer::KhatriRaoVectorizer) = iscompatible(rho,vectorizer) ? vecdp(rho) : internal_rep(Matrix(rho), vectorizer)
iscompatible(rho::BlockDiagonal, vectorizer::KhatriRaoVectorizer) = size.(rho.blocks, 1) == size.(rho.blocks, 2) == vectorizer.sizes 
_internal_vec(rho, vectorizer::KhatriRaoVectorizer)  = internal_rep(Matrix(rho), vectorizer)
function _internal_vec(rho::AbstractMatrix{T}, vectorizer::KhatriRaoVectorizer) where T
    inds = sizestoinds(vectorizer.sizes)
    indsv = sizestoinds(vectorizer.sizes .^2 )
    v = Vector{T}(undef, sum(vectorizer.sizes .^2))
    for i in eachindex(inds)
        v[indsv[i]] = vec(rho[inds[i],inds[i]])
    end
    return v
end

LinearOperator(system::LindbladSystem; kwargs...) = LinearOperator(system.lindblad; kwargs...)
LinearOperatorWithNormalizer(system::LindbladSystem; kwargs...) = LinearOperator(lindblad_with_normalizer(system.lindblad, system.vectorizer); kwargs...)

Base.show(io::IO, ::MIME"text/plain", system::LindbladSystem) = show(io, system)
Base.show(io::IO, system::LindbladSystem) = print(io, "LindbladSystem:", "\nOpenSystem", repr(system.system),
    "\nUnitary: ", reprlindblad(system.unitary), "\ndissipators: ", reprdissipators(system.dissipators),
    "\nlindblad: ", reprlindblad(system.lindblad),
    "\nvectorizer: ", typeof(system.vectorizer))

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
commutator(A, ::KronVectorizer) = commutator(A)
commutator(A) = kron(one(A), A) - kron(transpose(A), one(A))
measure(rho::AbstractMatrix, ls::LindbladSystem) = measure(internal_rep(rho,ls), ls)
measure(rho, ls::LindbladSystem) = map(op-> measure(rho, op, ls::LindbladSystem), transformed_measurements(ls))
measure(rho, op, ls::LindbladSystem) = map(d -> measure_dissipator(rho, op, d,ls), ls.dissipators)

function measure_dissipator(rho, op::AbstractMatrix, dissipator::NamedTuple{(:in, :out, :label),<:Any}, system)
    results = map(dissipator_op -> measure(rho, op, dissipator_op, system), (; dissipator.in, dissipator.out))
    merge(results, (; total=sum(results), label=dissipator.label))
end
measure(rho, op, dissipator, system::LindbladSystem) = dot(conj(op), dissipator * rho)
# measure(rho, op::AbstractMatrix, ::KronVectorizer, dissipator) = dot(conj(vec(op)), dissipator * vec(rho))
# measure(rho::BlockDiagonal, op::BlockDiagonal, ::KhatriRaoVectorizer, dissipator) = dot(conj(vecdp(op)), dissipator * vecdp(rho))
# measure(rho, op, dissipator) = dot(vec(op),dissipator*rho)
# current(ρ,op,sj) = tr(op * reshape(sj*ρ,size(op)))
# commutator(T1,T2) = -T1⊗T2 + T2⊗T1
# commutator(A) = -transpose(A)⊗one(A) + one(A)⊗A


lindblad_with_normalizer_dense(lindblad::Matrix, kv::AbstractVectorizer) = vcat(lindblad, transpose(kv.idvec))

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
lindblad_with_normalizer(lindblad::Matrix, vectorizer) = lindblad_with_normalizer_dense(lindblad, vectorizer)

_identity_density_matrix(system::LindbladSystem) = one(eltype(system.lindblad)) * (system.vectorizer.idvec ./ sqrt(size(system.lindblad, 2)))

function prepare_lindblad(system::OpenSystem; kwargs...)
    diagonalsystem = diagonalize(system; kwargs...)
    prepare_lindblad(diagonalsystem; kwargs...)
end
function prepare_lindblad(diagonalsystem::OpenSystem{<:DiagonalizedHamiltonian}; kwargs...)
    transformedsystem = ratetransform(diagonalsystem)
    vectorizer = default_vectorizer(hamiltonian(diagonalsystem))
    dissipators = map(lead -> dissipator_from_transformed_lead(lead, vectorizer), transformed_leads(transformedsystem))
    unitary = -1im * commutator(eigenvalues(transformedsystem), vectorizer)
    lindblad = unitary + sum(d -> d.in + d.out, dissipators)
    lindbladsystem = LindbladSystem(transformedsystem, unitary, dissipators, lindblad, vectorizer)
    return lindbladsystem
end

function dissipator_from_transformed_lead(lead::NormalLead, vectorizer::AbstractVectorizer)
    opin = dissipator(lead.jump_in, vectorizer)
    opout = dissipator(lead.jump_out, vectorizer)
    (; in=opin, out=opout, label=lead.label)
end
