module QuantumDotsBlockDiagonalsExt

using QuantumDots, BlockDiagonals, LinearAlgebra, SparseArrays
import QuantumDots: FockSymmetry, DiagonalizedHamiltonian, blockdiagonal, KhatriRaoVectorizer, vecdp


function diagonalize(m::BlockDiagonal)
    vals, vecs = BlockDiagonals.eigen_blockwise(m)
    DiagonalizedHamiltonian(vals, vecs, m)
end

function diagonalize!(m::BlockDiagonal; original=nothing)
    vals, vecs = eigen!_blockwise(m)
    DiagonalizedHamiltonian(vals, vecs, original)
end
diagonalize(m::BlockDiagonal{<:Any,<:SparseMatrixCSC}) = diagonalize!(m)
diagonalize!(m::BlockDiagonal{<:Any,<:SparseMatrixCSC}) = diagonalize!(BlockDiagonal(Matrix.(m.blocks)); original=m)
diagonalize(m::BlockDiagonal{<:Any,<:Hermitian{<:Any,<:SparseMatrixCSC}}) = diagonalize!(m)
diagonalize!(m::BlockDiagonal{<:Any,<:Hermitian{<:Any,<:SparseMatrixCSC}}) = diagonalize!(BlockDiagonal((Hermitian ∘ Matrix).(m.blocks)), original=m)

function eigen!_blockwise(B::BlockDiagonal, args...; kwargs...)
    eigens = [eigen!(b, args...; kwargs...) for b in blocks(B)]
    values = [e.values for e in eigens]
    vectors = [e.vectors for e in eigens]
    vcat(values...), BlockDiagonal(vectors)
end



"""
    blockdiagonal(m::AbstractMatrix, basis::SymmetricFockHilbertSpace)

Construct a BlockDiagonal version of `m` using the symmetry of `basis`. No checking is done to ensure this is a faithful representation.
"""
blockdiagonal(m::AbstractMatrix, basis::SymmetricFockHilbertSpace) = blockdiagonal(m, basis.symmetry)
blockdiagonal(::Type{T}, m::AbstractMatrix, basis::SymmetricFockHilbertSpace) where {T} = blockdiagonal(T, m, basis.symmetry)

blockdiagonal(m::AbstractMatrix, ::NoSymmetry) = m
function blockdiagonal(m::AbstractMatrix, sym::FockSymmetry)
    blockinds = values(sym.qntoinds)
    BlockDiagonal([m[block, block] for block in blockinds])
end
function blockdiagonal(::Type{T}, m::AbstractMatrix, sym::FockSymmetry) where {T}
    blockinds = values(sym.qntoinds)
    BlockDiagonal([T(m[block, block]) for block in blockinds])
end
function blockdiagonal(m::Hermitian, sym::FockSymmetry)
    blockinds = values(sym.qntoinds)
    Hermitian(BlockDiagonal([m[block, block] for block in blockinds]))
end
function blockdiagonal(::Type{T}, m::Hermitian, sym::FockSymmetry) where {T}
    blockinds = values(sym.qntoinds)
    Hermitian(BlockDiagonal([T(m[block, block]) for block in blockinds]))
end

_blocks(m::AbstractMatrix) = [m]
_blocks(m::BlockDiagonal) = blocks(m)
function BlockDiagonals.blocks(eig::DiagonalizedHamiltonian; full=false)
    vals = eig.values
    vecs = eig.vectors
    bvecs = _blocks(vecs)
    sizes = size.(bvecs, 1)
    blockinds = map(i -> eachindex(vals)[i], sizestoinds(sizes))
    if full
        filteredinds = [map(i -> i in inds, eachindex(vals)) for inds in blockinds]
        map((inds, block) -> DiagonalizedHamiltonian(vals[inds], vecs[:, inds], original_hamiltonian(eig)), filteredinds, bvecs)
    else
        original_blocks = _blocks(original_hamiltonian(eig))
        map((inds, block, hblock) -> DiagonalizedHamiltonian(vals[inds], block, hblock), blockinds, bvecs, original_blocks)
    end
end

khatri_rao(L1::BlockDiagonal, L2::BlockDiagonal) = cat([kron(B1, B2) for (B1, B2) in zip(blocks(L1), blocks(L2))]...; dims=(1, 2))
function khatri_rao(L1::BlockDiagonal, L2::BlockDiagonal, kv::KhatriRaoVectorizer)
    if kv.sizes == first.(blocksizes(L1)) == first.(blocksizes(L2)) == last.(blocksizes(L1)) == last.(blocksizes(L2))
        return khatri_rao(L1, L2)
    else
        return khatri_rao(cat(L1.blocks...; dims=(1, 2)), cat(L2.blocks...; dims=(1, 2)), kv)
    end
end

khatri_rao_commutator(A::BlockDiagonal{<:Any,<:Diagonal}, blocksizes) = khatri_rao_commutator(Diagonal(A), blocksizes)

kr_one(m::BlockDiagonal) = BlockDiagonal(kr_one.(blocks(m)))


internal_rep(rho::BlockDiagonal, vectorizer::KhatriRaoVectorizer) = iscompatible(rho, vectorizer) ? vecdp(rho) : internal_rep(Matrix(rho), vectorizer)
iscompatible(rho::BlockDiagonal, vectorizer::KhatriRaoVectorizer) = size.(rho.blocks, 1) == size.(rho.blocks, 2) == vectorizer.sizes

default_vectorizer(ham::BlockDiagonal) = KhatriRaoVectorizer(ham)
vecdp(bd::BlockDiagonal) = mapreduce(vec, vcat, blocks(bd))

function remove_high_energy_states(ham::DiagonalizedHamiltonian{<:Any,<:BlockDiagonal}, ΔE)
    E0 = minimum(eigenvalues(ham))
    sectors = blocks(ham)
    Is = map(eig -> findall(<(ΔE + E0), eig.values), sectors)
    newblocks = map((eig, I) -> eig.vectors[:, I], sectors, Is)
    newvals = map((eig, I) -> eig.values[I], sectors, Is)
    E = reduce(vcat, newvals)
    vecs = BlockDiagonal(newblocks)
    DiagonalizedHamiltonian(E, vecs, original_hamiltonian(E, vecs))
end

end
