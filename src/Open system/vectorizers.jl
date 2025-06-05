
abstract type AbstractVectorizer end
"""
    struct KronVectorizer{T} <: AbstractVectorizer

A struct representing a KronVectorizer, the standard vectorizer where superoperators are formed from kronecker products of operators.

# Fields
- `size::Int`: The size of the KronVectorizer.
- `idvec::Vector{T}`: A vector representing the vectorized identity matrix. Saved here because it is useful when normalization is needed for computations in e.g. LindbladSystem.
"""
struct KronVectorizer{T} <: AbstractVectorizer
    size::Int
    idvec::Vector{T}
end
KronVectorizer(n::Integer, ::Type{T}=Float64) where {T} = KronVectorizer{T}(n, vec(Matrix{T}(I, n, n)))

"""
    struct KhatriRaoVectorizer{T} <: AbstractVectorizer

A struct representing a Khatri-Rao vectorizer. This vectorizer is used for BlockDiagonal density matrices matrices, where the superoperators respect the block structure.

# Fields
- `sizes::Vector{Int}`: Vector of sizes.
- `idvec::Vector{T}`: Vector of identifiers.
- `cumsum::Vector{Int}`: Vector of cumulative sums.
- `cumsumsquared::Vector{Int}`: Vector of squared cumulative sums.
- `inds::Vector{UnitRange{Int}}`: Vector of index ranges.
- `vectorinds::Vector{UnitRange{Int}}`: Vector of vector index ranges.
"""
struct KhatriRaoVectorizer{T} <: AbstractVectorizer
    sizes::Vector{Int}
    idvec::Vector{T}
    cumsum::Vector{Int}
    cumsumsquared::Vector{Int}
    inds::Vector{UnitRange{Int}}
    vectorinds::Vector{UnitRange{Int}}
    linearindices::Matrix{Matrix{Int}}
end
function KhatriRaoVectorizer(sizes::Vector{Int}, ::Type{T}=Float64) where {T}
    blockid = BlockDiagonal([Matrix{T}(I, size, size) for size in sizes])
    vectorinds = sizestoinds(sizes .^ 2)
    linearindices = [LinearIndices((sum(sizes .^ 2), sum(sizes .^ 2)))[i1, i2] for i1 in vectorinds, i2 in vectorinds]
    KhatriRaoVectorizer{T}(sizes, vecdp(blockid), [0, cumsum(sizes)...], [0, cumsum(sizes .^ 2)...], sizestoinds(sizes), vectorinds, linearindices)
end

KronVectorizer(ham) = KronVectorizer(size(ham, 1), eltype(ham))
KhatriRaoVectorizer(ham) = KhatriRaoVectorizer(first.(blocksizes(ham)), eltype(ham))

default_vectorizer(ham::BlockDiagonal) = KhatriRaoVectorizer(ham)
default_vectorizer(ham) = KronVectorizer(ham)
