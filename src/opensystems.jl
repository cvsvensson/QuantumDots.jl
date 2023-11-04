fermidirac(E, T, μ) = (I + exp((E - μ * I) / T))^(-1)
abstract type AbstractLead end
struct NormalLead{W1,W2,Opin,Opout} <: AbstractLead
    T::W1
    μ::W2
    jump_in::Opin
    jump_out::Opout
end
NormalLead(jin, jout; T, μ) = NormalLead(T, μ, [jin], [jout])
NormalLead(jin; T, μ) = NormalLead(jin, jin'; T, μ)
function update_lead(lead, props)
    μ = get(props, :μ, lead.μ)
    T = get(props, :T, lead.T)
    NormalLead(T, μ, lead.jump_in, lead.jump_out)
end
CombinedLead(jins; T, μ) = CombinedLead(jins, map(adjoint, jins); T, μ)
CombinedLead(jins, jouts; T, μ) = NormalLead(T, μ, jins, jouts)

Base.adjoint(lead::NormalLead) = NormalLead(lead.T, lead.μ, map(adjoint, lead.jump_in), map(adjoint, lead.jump_out))

Base.show(io::IO, ::MIME"text/plain", lead::NormalLead{T,Opin,Opout,N}) where {T,Opin,Opout,N} = print(io, "NormalLead{$T,$Opin,$Opout,$N}(T=", temperature(lead), ", μ=", chemical_potential(lead), ")")
Base.show(io::IO, lead::NormalLead{T,Opin,Opout,N}) where {T,Opin,Opout,N} = print(io, "Lead(, T=", round(temperature(lead), digits=4), ", μ=", round(chemical_potential(lead), digits=4), ")")

chemical_potential(lead::NormalLead) = lead.μ
temperature(lead::NormalLead) = lead.T

abstract type AbstractDissipator end
Base.:*(d::AbstractDissipator, v) = Matrix(d) * v
LinearAlgebra.mul!(v, d::AbstractDissipator, u) = mul!(v, Matrix(d), u)
LinearAlgebra.mul!(v, d::AbstractDissipator, u, a, b) = mul!(v, Matrix(d), u, a, b)
Base.size(d::AbstractDissipator, i) = size(Matrix(d), i)
Base.size(d::AbstractDissipator) = size(Matrix(d))
Base.eltype(d::AbstractDissipator) = eltype(Matrix(d))
SciMLBase.islinear(d::AbstractDissipator) = true

##
struct DiagonalizedHamiltonian{Vals,Vecs,H} <: AbstractDiagonalHamiltonian
    values::Vals
    vectors::Vecs
    original::H
end
Base.eltype(::DiagonalizedHamiltonian{Vals,Vecs}) where {Vals,Vecs} = promote_type(eltype(Vals), eltype(Vecs))
Base.size(h::DiagonalizedHamiltonian) = size(eigenvectors(h))
Base.:-(h::DiagonalizedHamiltonian) = DiagonalizedHamiltonian(-h.values, -h.vectors, -h.original)
Base.iterate(S::DiagonalizedHamiltonian) = (S.values, Val(:vectors))
Base.iterate(S::DiagonalizedHamiltonian, ::Val{:vectors}) = (S.vectors, Val(:original))
Base.iterate(S::DiagonalizedHamiltonian, ::Val{:original}) = (S.original, Val(:done))
Base.iterate(::DiagonalizedHamiltonian, ::Val{:done}) = nothing
Base.adjoint(H::DiagonalizedHamiltonian) = DiagonalizedHamiltonian(conj(H.values), adjoint(H.vectors), adjoint(H.original))
original_hamiltonian(H::DiagonalizedHamiltonian) = H.original

abstract type AbstractOpenSystem end
Base.:*(d::AbstractOpenSystem, v) = Matrix(d) * v
LinearAlgebra.mul!(v, d::AbstractOpenSystem, u) = mul!(v, Matrix(d), u)
LinearAlgebra.mul!(v, d::AbstractOpenSystem, u, a, b) = mul!(v, Matrix(d), u, a, b)
Base.eltype(system::AbstractOpenSystem) = eltype(Matrix(system))
Base.size(d::AbstractOpenSystem, i) = size(Matrix(d), i)
Base.size(d::AbstractOpenSystem) = size(Matrix(d))
SciMLBase.islinear(d::AbstractOpenSystem) = true


abstract type AbstractOpenSolver end

function normalized_steady_state_rhs(A)
    n = size(A, 2)
    b = zeros(eltype(A), n)
    push!(b, one(eltype(A)))
    return b
end


function StationaryStateProblem(system::AbstractOpenSystem, p=SciMLBase.NullParameters(); u0=identity_density_matrix(system), kwargs...)
    A = LinearOperator(system, p; normalizer=true, kwargs...)
    b = normalized_steady_state_rhs(A)
    LinearProblem(A, b; u0, kwargs...)
end
function ODEProblem(system::AbstractOpenSystem, u0, tspan, p=SciMLBase.NullParameters(), args...; kwargs...)
    internalu0 = internal_rep(u0, system)
    _ODEProblem(system, internalu0, tspan, p, args...; kwargs...)
end
function _ODEProblem(system::AbstractOpenSystem, u0, tspan, p, args...; kwargs...)
    ODEProblem(LinearOperator(system, p; kwargs...), u0, tspan, p, args...; kwargs...)
end

function solveDiffProblem!(linsolve, x0, dA)
    linsolve.b[1:end-1] .= -dA * x0
    linsolve.b[end] = zero(eltype(linsolve.b))
    return solve!(linsolve)
end

LinearOperator(mat::AbstractMatrix; kwargs...) = MatrixOperator(mat; kwargs...)

function changebasis(lead::NormalLead, H::DiagonalizedHamiltonian)
    S = eigenvectors(H)
    NormalLead(temperature(lead), chemical_potential(lead), map(op -> S' * op * S, lead.jump_in), map(op -> S' * op * S, lead.jump_out))
end

trnorm(rho, n) = tr(reshape(rho, n, n))
vecdp(bd::BlockDiagonal) = mapreduce(vec, vcat, blocks(bd))
original_hamiltonian(E, vecs) = vecs * Diagonal(E) * vecs'
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
function remove_high_energy_states(ham::DiagonalizedHamiltonian, ΔE)
    vals = eigenvalues(ham)
    vecs = eigenvectors(ham)
    E0 = minimum(vals)
    I = findall(<(ΔE + E0), vals)
    newvecs = vecs[:, I]
    newvals = vals[I]
    DiagonalizedHamiltonian(newvals, newvecs, original_hamiltonian(newvals, newvecs))
end


function ratetransform(op, diagham::DiagonalizedHamiltonian, T, μ)
    op2 = changebasis(op, diagham)
    op3 = ratetransform(op2, diagham.values, T, μ)
    return changebasis(op3, diagham')
end
ratetransform(op, energies::AbstractVector, T, μ) = reshape(sqrt(fermidirac(commutator(Diagonal(energies)), T, μ)) * vec(op), size(op))