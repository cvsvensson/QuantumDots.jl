
abstract type AbstractVectorizer end
struct KronVectorizer{T} <: AbstractVectorizer
    size::Int
    idvec::Vector{T}
end
KronVectorizer(n::Integer, ::Type{T}=Float64) where {T} = KronVectorizer{T}(n, vec(Matrix{T}(I, n, n)))

struct KhatriRaoVectorizer{T} <: AbstractVectorizer
    sizes::Vector{Int}
    idvec::Vector{T}
    cumsum::Vector{Int}
    cumsumsquared::Vector{Int}
    inds::Vector{UnitRange{Int}}
    vectorinds::Vector{UnitRange{Int}}
end
function KhatriRaoVectorizer(sizes::Vector{Int}, ::Type{T}=Float64) where {T}
    blockid = BlockDiagonal([Matrix{T}(I, size, size) for size in sizes])
    KhatriRaoVectorizer{T}(sizes, vecdp(blockid), [0, cumsum(sizes)...], [0, cumsum(sizes .^2)...], sizestoinds(sizes), sizestoinds(sizes .^2))
end

KronVectorizer(ham) = KronVectorizer(size(ham,1), eltype(ham))
KhatriRaoVectorizer(ham) = KhatriRaoVectorizer(first.(blocksizes(ham)), eltype(ham))

default_vectorizer(ham::BlockDiagonal) = KhatriRaoVectorizer(ham)
default_vectorizer(ham) = KronVectorizer(ham)
