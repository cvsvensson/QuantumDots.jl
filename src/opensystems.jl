fermidirac(E, T, μ) = (I + exp(E / T)exp(-μ / T))^(-1)
abstract type AbstractLead end
struct NormalLead{W,Opin,Opout,L} <: AbstractLead
    T::W
    μ::W
    jump_in::Opin
    jump_out::Opout
    label::L
end
NormalLead(T, μ, in, out, label) = NormalLead(promote(T, μ)..., in, out, label)
NormalLead(jin, jout; T, μ, label=missing) = NormalLead(T, μ, jin, jout, label)
NormalLead(jin; T, μ, label=missing) = NormalLead(T, μ, jin, jin', label)
NormalLead(l::NormalLead; T=temperature(l), μ=chemical_potential(l), label=l.label, in=l.jump_in, out=l.jump_out) = NormalLead(T, μ, in, out, label)

Base.show(io::IO, ::MIME"text/plain", lead::NormalLead{T,Opin,Opout,N}) where {T,Opin,Opout,N} = print(io, "NormalLead{$T,$Opin,$Opout,$N}(Label=", lead.label, ", T=", temperature(lead), ", μ=", chemical_potential(lead), ")")
Base.show(io::IO, lead::NormalLead{T,Opin,Opout,N}) where {T,Opin,Opout,N} = print(io, "Lead(", lead.label, ", T=", round(temperature(lead), digits=4), ", μ=", round(chemical_potential(lead), digits=4), ")")

chemical_potential(lead::NormalLead) = lead.μ
temperature(lead::NormalLead) = lead.T

##
struct DiagonalizedHamiltonian{Vals,Vecs}
    eigenvalues::Vals
    eigenvectors::Vecs
end
Base.eltype(::DiagonalizedHamiltonian{Vals,Vecs}) where {Vals,Vecs} = promote_type(eltype(Vals), eltype(Vecs))

abstract type AbstractOpenSystem end
struct OpenSystem{H,L,M1,M2} <: AbstractOpenSystem
    hamiltonian::H
    leads::L
    measurements::M1
    transformed_measurements::M2
end
OpenSystem(H) = OpenSystem(H, nothing, nothing, nothing)
OpenSystem(H, l) = OpenSystem(H, l, nothing, nothing)
OpenSystem(H, l, m) = OpenSystem(H, l, m, nothing)
eigenvaluevector(H::OpenSystem{<:DiagonalizedHamiltonian}) = diag(eigenvalues(H))
hamiltonian(system::OpenSystem) = system.hamiltonian
eigenvalues(system::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvalues(hamiltonian(system))
eigenvectors(system::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvectors(hamiltonian(system))
eigenvalues(hamiltonian::DiagonalizedHamiltonian) = hamiltonian.eigenvalues
eigenvectors(hamiltonian::DiagonalizedHamiltonian) = hamiltonian.eigenvectors
leads(system::OpenSystem) = system.leads
measurements(system::OpenSystem) = system.measurements
transformed_measurements(system::OpenSystem) = system.transformed_measurements
changebasis(op, os::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvectors(os)' * op * eigenvectors(os)
changebasis(::Nothing, os::OpenSystem{<:DiagonalizedHamiltonian}) = nothing

Base.show(io::IO, ::MIME"text/plain", system::OpenSystem) = show(io, system)
Base.show(io::IO, system::OpenSystem{H,L,M1,M2}) where {H,L,M1,M2} = print(io, "OpenSystem:\nHamiltonian: ", repr(system.hamiltonian), "\nleads: ", repr(system.leads),  "\nmeasurements: ", repr(measurements(system)), "\ntransformed_measurements: ", repr(transformed_measurements(system)))

abstract type AbstractOpenSolver end

function normalized_steady_state_rhs(A)
    n = size(A, 2)
    b = zeros(eltype(A), n)
    push!(b, one(eltype(A)))
    return b
end

function LinearProblem(system::AbstractOpenSystem; kwargs...)
    prob = _LinearProblem(system; kwargs...)
end
function _LinearProblem(system::AbstractOpenSystem, args...; kwargs...)
    A = LinearOperatorWithNormalizer(system; kwargs...)
    b = normalized_steady_state_rhs(A)
    u0 = identity_density_matrix(system)
    lp = LinearProblem(A, b; u0, kwargs...)
end
function ODEProblem(system::AbstractOpenSystem, u0, args...; kwargs...)
    internalu0 = internal_rep(u0, system)
    prob = _ODEProblem(system, internalu0, args...; kwargs...)
end
function _ODEProblem(system::AbstractOpenSystem, u0, args...; kwargs...)
    op = ODEProblem(LinearOperator(system; kwargs...), u0, args...; kwargs...)
end

# function differentiate!(linsolve::LinearSolve.LinearCache, x0, dA)
#     linsolve.b = -dA * x0
#     solve!(linsolve)
# end

LinearOperator(mat::AbstractMatrix; kwargs...) = MatrixOperator(mat; kwargs...)
LinearOperator(func::Function; kwargs...) = FunctionOperator(func; islinear=true, kwargs...)

diagonalize(S, lead::NormalLead) = NormalLead(temperature(lead), chemical_potential(lead), S' * lead.jump_in * S, S' * lead.jump_out * S, lead.label)
diagonalize_hamiltonian(system::OpenSystem) = OpenSystem(diagonalize(hamiltonian(system)), leads(system), measurements(system), transformed_measurements(system))

function diagonalize(m::AbstractMatrix)
    vals, vecs = eigen(m)
    DiagonalizedHamiltonian(Diagonal(vals), vecs)
end
diagonalize(m::SparseMatrixCSC) = diagonalize(Matrix(m))
function diagonalize(m::BlockDiagonal)
    vals, vecs = BlockDiagonals.eigen_blockwise(m)
    blockinds = sizestoinds(map(first, blocksizes(vecs)))
    bdvals = BlockDiagonal(map(inds -> Diagonal(vals[inds]), blockinds))
    DiagonalizedHamiltonian(bdvals, vecs)
end
diagonalize(m::BlockDiagonal{<:Any,<:SparseMatrixCSC}) = diagonalize(BlockDiagonal(Matrix.(m.blocks)))
diagonalize(m::BlockDiagonal{<:Any,<:Hermitian{<:Any,<:SparseMatrixCSC}}) = diagonalize(BlockDiagonal(Hermitian.(Matrix.(m.blocks))))

diagonalize_leads(system::OpenSystem{<:DiagonalizedHamiltonian}) = OpenSystem(hamiltonian(system), map(lead-> diagonalize(eigenvectors(system), lead), leads(system)), measurements(system), transformed_measurements(system))
transform_measurements(system::OpenSystem{<:DiagonalizedHamiltonian}) = OpenSystem(hamiltonian(system), leads(system),  measurements(system), map(op -> changebasis(op, system), measurements(system)))
transform_measurements(system::OpenSystem{<:DiagonalizedHamiltonian,<:Any,Nothing}) = system

function diagonalize(system::OpenSystem; dE=0.0)
    diagonal_system = diagonalize_hamiltonian(system)
    if dE > 0
        diagonal_system = remove_high_energy_states(dE, diagonal_system)
    end
    transform_measurements(diagonalize_leads(diagonal_system))
end

trnorm(rho, n) = tr(reshape(rho, n, n))
vecdp(bd::BlockDiagonal) = mapreduce(vec, vcat, blocks(bd))

remove_high_energy_states(dE, system::OpenSystem) = OpenSystem(remove_high_energy_states(dE, hamiltonian(system)), leads(system), measurements(system), transformed_measurements(system))
function remove_high_energy_states(ΔE, ham::DiagonalizedHamiltonian{<:BlockDiagonal,<:BlockDiagonal})
    vals = eigenvalues(ham)
    vecs = eigenvectors(ham)
    E0 = minimum(diag(vals))
    Is = map(vals -> findall(<(ΔE + E0), diag(vals)), blocks(vals))
    newblocks = map((block, I) -> block[:, I], blocks(vecs), Is)
    newvals = map((vals, I) -> Diagonal(diag(vals)[I]), blocks(vals), Is)
    DiagonalizedHamiltonian(BlockDiagonal(newvals), BlockDiagonal(newblocks))
end
function remove_high_energy_states(ΔE, ham::DiagonalizedHamiltonian)
    vals = eigenvalues(ham)
    vecs = eigenvectors(ham)
    E0 = minimum(diag(vals))
    I = findall(<(ΔE + E0), diag(vals))
    newvecs = vecs[:, I]
    newvals = Diagonal(diag(vals)[I])
    DiagonalizedHamiltonian(newvals, newvecs)
end

stationary_state(system::AbstractOpenSystem, alg=nothing; kwargs...) = solve(LinearProblem(system), alg; kwargs...)
stationary_state(method::AbstractOpenSolver, system::OpenSystem, alg=nothing; kwargs...) = solve(LinearProblem(method, system), alg; kwargs...)

function LinearProblem(method::AbstractOpenSolver, H::AbstractMatrix, leads, measurements=nothing; kwargs...)
    LinearProblem(method, OpenSystem(H, leads, nothing, measurements, nothing); kwargs...)
end


# function ratetransform(system::OpenSystem{<:DiagonalizedHamiltonian})
#     comm = commutator(Diagonal(eigenvalues(system)))
#     newleads = [ratetransform(lead, comm) for lead in leads(system)]
#     return OpenSystem(hamiltonian(system), leads(system), newleads, measurements(system), transformed_measurements(system))
# end


# function ratetransform(lead::NormalLead, commutator_hamiltonian)
#     μ = chemical_potential(lead)
#     T = temperature(lead)
#     newjumpin = ratetransform(lead.jump_in, commutator_hamiltonian, T, μ) 
#     newjumpout = ratetransform(lead.jump_out, commutator_hamiltonian, T, -μ)
#     return NormalLead(T, μ, newjumpin, newjumpout, lead.label)
# end
ratetransform(op, commutator_hamiltonian::Diagonal, T, μ) = ratetransform!(zero(op),op, commutator_hamiltonian,T,μ)
function ratetransform!(op2,op, commutator_hamiltonian::Diagonal, T, μ) 
    mul!(vec(op2), sqrt(fermidirac(commutator_hamiltonian, T, μ)), vec(op))
    return op2
end
