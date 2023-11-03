hamiltonian(system::OpenSystem) = system.hamiltonian
eigenvalues(system::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvalues(hamiltonian(system))
eigenvectors(system::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvectors(hamiltonian(system))
eigenvalues(H::DiagonalizedHamiltonian) = H.values
eigenvectors(H::DiagonalizedHamiltonian) = H.vectors
diagonalize(eig::DiagonalizedHamiltonian) = eig
diagonalize!(eig::DiagonalizedHamiltonian) = eig

function diagonalize(m::AbstractMatrix)
    vals, vecs = eigen(m)
    DiagonalizedHamiltonian(vals, vecs)
end
diagonalize(m::SparseMatrixCSC) = diagonalize!(Matrix(m))
diagonalize(m::Hermitian{<:Any,<:SparseMatrixCSC}) = diagonalize!(Hermitian(Matrix(m)))
function diagonalize(m::BlockDiagonal)
    vals, vecs = BlockDiagonals.eigen_blockwise(m)
    DiagonalizedHamiltonian(vals, vecs, m)
end
function diagonalize!(m::AbstractMatrix)
    vals, vecs = eigen!(m)
    DiagonalizedHamiltonian(vals, vecs, m)
end
diagonalize!(m::SparseMatrixCSC) = diagonalize!(Matrix(m))
diagonalize!(m::Hermitian{<:Any,<:SparseMatrixCSC}) = diagonalize!(Hermitian(Matrix(m)))
function diagonalize!(m::BlockDiagonal)
    vals, vecs = eigen!_blockwise(m)
    DiagonalizedHamiltonian(vals, vecs, nothing)
end
diagonalize(m::BlockDiagonal{<:Any,<:SparseMatrixCSC}) = diagonalize!(m)
diagonalize!(m::BlockDiagonal{<:Any,<:SparseMatrixCSC}) = diagonalize!(BlockDiagonal(Matrix.(m.blocks)))
diagonalize(m::BlockDiagonal{<:Any,<:Hermitian{<:Any,<:SparseMatrixCSC}}) = diagonalize!(m)
diagonalize!(m::BlockDiagonal{<:Any,<:Hermitian{<:Any,<:SparseMatrixCSC}}) = diagonalize!(BlockDiagonal((Hermitian ∘ Matrix).(m.blocks)))

function eigen!_blockwise(B::BlockDiagonal, args...; kwargs...)
    eigens = [eigen!(b, args...; kwargs...) for b in blocks(B)]
    values = [e.values for e in eigens]
    vectors = [e.vectors for e in eigens]
    vcat(values...), BlockDiagonal(vectors)
end
BlockDiagonals.blocks(m::AbstractMatrix) = [m]

function BlockDiagonals.blocks(eig::DiagonalizedHamiltonian; full=false)
    vals = eig.values
    vecs = eig.vectors
    bvecs = blocks(vecs)
    sizes = size.(bvecs, 1)
    blockinds = map(i -> eachindex(vals)[i], sizestoinds(sizes))
    if full
        filteredinds = [map(i -> i in inds, eachindex(vals)) for inds in blockinds]
        map((inds, block) -> DiagonalizedHamiltonian(vals[inds], vecs[:, inds]), filteredinds, bvecs)
    else
        map((inds, block) -> DiagonalizedHamiltonian(vals[inds], block), blockinds, bvecs)
    end
end

function ground_state(eig::DiagonalizedHamiltonian)
    vals = eig.values
    vecs = eig.vectors
    minind = argmin(vals)
    (;value = vals[minind], vector = vecs[:, minind])
end
