
abstract type AbstractVectorizer end
struct KronVectorizer{T} <: AbstractVectorizer
    size::Int
    idvec::Vector{T}
end
KronVectorizer(n::Integer, ::Type{T}=Float64) where {T} = KronVectorizer{T}(n, vec(Matrix{T}(I, n, n)))

struct KhatriRaoVectorizer{T} <: AbstractVectorizer
    sizes::Vector{Int}
    idvec::Vector{T}
end
function KhatriRaoVectorizer(sizes::Vector{Int}, ::Type{T}=Float64) where {T}
    blockid = BlockDiagonal([Matrix{T}(I, size, size) for size in sizes])
    KhatriRaoVectorizer{T}(sizes, vecdp(blockid))
end

KronVectorizer(ham::DiagonalizedHamiltonian) = KronVectorizer(size(ham.eigenvalues, 1), eltype(ham))
KhatriRaoVectorizer(ham::DiagonalizedHamiltonian) = KhatriRaoVectorizer(first.(blocksizes(ham.eigenvalues)), eltype(ham))

default_vectorizer(ham::DiagonalizedHamiltonian{<:BlockDiagonal}) = KhatriRaoVectorizer(ham)
default_vectorizer(ham::DiagonalizedHamiltonian) = KronVectorizer(ham)
