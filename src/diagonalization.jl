eigenvalues(H::DiagonalizedHamiltonian) = H.values
eigenvectors(H::DiagonalizedHamiltonian) = H.vectors
diagonalize(eig::DiagonalizedHamiltonian) = eig
diagonalize!(eig::DiagonalizedHamiltonian) = eig

function diagonalize(m::AbstractMatrix)
    vals, vecs = eigen(m)
    DiagonalizedHamiltonian(vals, vecs, m)
end
diagonalize(m::SparseMatrixCSC) = diagonalize!(Matrix(m), original=m)
diagonalize(m::Hermitian{<:Any,<:SparseMatrixCSC}) = diagonalize!(Hermitian(Matrix(m)), original=m)
function diagonalize(m::BlockDiagonal)
    vals, vecs = BlockDiagonals.eigen_blockwise(m)
    DiagonalizedHamiltonian(vals, vecs, m)
end
function diagonalize!(m::AbstractMatrix; original=nothing)
    vals, vecs = eigen!(m)
    DiagonalizedHamiltonian(vals, vecs, original)
end
diagonalize!(m::SparseMatrixCSC) = diagonalize!(Matrix(m); original=m)
diagonalize!(m::Hermitian{<:Any,<:SparseMatrixCSC}) = diagonalize!(Hermitian(Matrix(m)); original=m)
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

function ground_state(eig::DiagonalizedHamiltonian)
    vals = eig.values
    vecs = eig.vectors
    minind = argmin(vals)
    (; value=vals[minind], vector=vecs[:, minind])
end

changebasis(op, os::DiagonalizedHamiltonian) = eigenvectors(os)' * op * eigenvectors(os)
changebasis(::Nothing, os::DiagonalizedHamiltonian) = nothing
function changebasis!(out, cache, op, os::DiagonalizedHamiltonian)
    mul!(cache, eigenvectors(os)', op)
    mul!(out, cache, eigenvectors(os))
end
function changebasis!(op, cache, os::DiagonalizedHamiltonian)
    mul!(cache, eigenvectors(os)', op)
    mul!(op, cache, eigenvectors(os))
end