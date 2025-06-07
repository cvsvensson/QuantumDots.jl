eigenvalues(H::DiagonalizedHamiltonian) = H.values
eigvals(H::DiagonalizedHamiltonian) = H.values
eigenvectors(H::DiagonalizedHamiltonian) = H.vectors
eigvecs(H::DiagonalizedHamiltonian) = H.vectors
diagonalize(eig::DiagonalizedHamiltonian) = eig
diagonalize!(eig::DiagonalizedHamiltonian) = eig

function diagonalize(m::AbstractMatrix)
    vals, vecs = eigen(m)
    DiagonalizedHamiltonian(vals, vecs, m)
end
diagonalize(m::SparseMatrixCSC) = diagonalize!(Matrix(m), original=m)
diagonalize(m::Hermitian{<:Any,<:SparseMatrixCSC}) = diagonalize!(Hermitian(Matrix(m)), original=m)

function diagonalize!(m::AbstractMatrix; original=nothing)
    vals, vecs = eigen!(m)
    DiagonalizedHamiltonian(vals, vecs, original)
end
diagonalize!(m::SparseMatrixCSC) = diagonalize!(Matrix(m); original=m)
diagonalize!(m::Hermitian{<:Any,<:SparseMatrixCSC}) = diagonalize!(Hermitian(Matrix(m)); original=m)


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