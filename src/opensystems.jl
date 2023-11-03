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
update_lead(l::NormalLead; T=temperature(l), μ=chemical_potential(l), in=l.jump_in, out=l.jump_out) = NormalLead(T, μ, in, out)
function update_lead(lead, props)
    μ = get(props, :μ, lead.μ)
    T = get(props, :T, lead.T)
    update_lead(lead; μ, T)
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
Base.iterate(S::DiagonalizedHamiltonian, ::Val{:vectors}) = (S.vectors, Val(:done))
Base.iterate(S::DiagonalizedHamiltonian, ::Val{:done}) = nothing
Base.adjoint(H::DiagonalizedHamiltonian) = DiagonalizedHamiltonian(conj(H.values), adjoint(H.vectors), adjoint(H.original))

abstract type AbstractOpenSystem end
Base.:*(d::AbstractOpenSystem, v) = Matrix(d) * v
LinearAlgebra.mul!(v, d::AbstractOpenSystem, u) = mul!(v, Matrix(d), u)
LinearAlgebra.mul!(v, d::AbstractOpenSystem, u, a, b) = mul!(v, Matrix(d), u, a, b)
Base.eltype(system::AbstractOpenSystem) = eltype(Matrix(system))
Base.size(d::AbstractOpenSystem, i) = size(Matrix(d), i)
Base.size(d::AbstractOpenSystem) = size(Matrix(d))
SciMLBase.islinear(d::AbstractOpenSystem) = true


struct OpenSystem{H,L} <: AbstractOpenSystem
    hamiltonian::H
    leads::L
    # diagonal_hamiltonian::HD
    # diagonal_leads::LD
end
Base.eltype(system::OpenSystem) = eltype(eigenvectors(system))
# OpenSystem(H) = OpenSystem(H, nothing, nothing,nothing,nothing)
# OpenSystem(H, l) = OpenSystem(H, nothing, l)
# OpenSystem(H, HD::AbstractDiagonalHamiltonian) = OpenSystem(H, HD, nothing)

leads(system::OpenSystem) = system.leads
changebasis(op, os::DiagonalizedHamiltonian) = eigenvectors(os)' * op * eigenvectors(os)
changebasis(::Nothing, os::DiagonalizedHamiltonian) = nothing

Base.show(io::IO, ::MIME"text/plain", system::OpenSystem) = show(io, system)
Base.show(io::IO, system::OpenSystem{H,L}) where {H,L} = print(io, "OpenSystem:\nHamiltonian: ", repr(system.hamiltonian), "\nleads: ", repr(system.leads))

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
    println(size(internalu0))
    prob = _ODEProblem(system, internalu0, tspan, p, args...; kwargs...)
end
function _ODEProblem(system::AbstractOpenSystem, u0, tspan, p, args...; kwargs...)
    op = ODEProblem(LinearOperator(system, p; kwargs...), u0, tspan, p, args...; kwargs...)
end

function solveDiffProblem!(linsolve, x0, dA)
    linsolve.b[1:end-1] .= -dA * x0
    linsolve.b[end] = zero(eltype(linsolve.b))
    return solve!(linsolve)
end

LinearOperator(mat::AbstractMatrix; kwargs...) = MatrixOperator(mat; kwargs...)
# LinearOperator(func::Function; kwargs...) = FunctionOperator(func; islinear=true, kwargs...)

# diagonalize(S, lead::NormalLead) = NormalLead(temperature(lead), chemical_potential(lead), map(op -> S' * op * S, lead.jump_in), map(op -> S' * op * S, lead.jump_out))
function diagonalize(system::OpenSystem; dE=0)
    diagham = diagonalize(system.hamiltonian)
    if dE > 0
        diagham = remove_high_energy_states(dE, diagham)
    end
    # diagleads = map(lead -> diagonalize(eigenvectors(diagham), lead), leads(system))
    OpenSystem(diagham, leads)
end

trnorm(rho, n) = tr(reshape(rho, n, n))
vecdp(bd::BlockDiagonal) = mapreduce(vec, vcat, blocks(bd))
original_hamiltonian(E, vecs) = vecs * Diagonal(E) * vecs'
function remove_high_energy_states(ΔE, ham::DiagonalizedHamiltonian{<:Any,<:BlockDiagonal})
    E0 = minimum(eigenvalues(ham))
    sectors = blocks(ham)
    Is = map(eig -> findall(<(ΔE + E0), eig.values), sectors)
    newblocks = map((eig, I) -> eig.vectors[:, I], sectors, Is)
    newvals = map((eig, I) -> eig.values[I], sectors, Is)
    E = reduce(vcat, newvals)
    vecs = BlockDiagonal(newblocks)
    DiagonalizedHamiltonian(E, vecs, original_hamiltonian(E, vecs))
end
function remove_high_energy_states(ΔE, ham::DiagonalizedHamiltonian)
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

# function ratetransform!(op2, op, energies::AbstractVector, T, μ)
#     for I in CartesianIndices(op)
#         n1, n2 = Tuple(I)
#         δE = energies[n1] - energies[n2]
#         op2[n1, n2] = sqrt(fermidirac(δE, T, μ)) * op[n1, n2]
#     end
#     return op2
# end

function conductance_matrix(current_op, ls::AbstractOpenSystem, args...)
    rho = solve(StationaryStateProblem(ls))
    conductance_matrix(rho, current_op, ls::AbstractOpenSystem, args...)
end

function conductance_matrix(rho, current_op, ls::AbstractOpenSystem, dμ)
    perturbations = map(d -> (; μ=d.lead.μ + dμ), ls.dissipators)
    function get_current(pert)
        newls = update(ls, pert)
        sol = solve(StationaryStateProblem(newls))
        collect(measure(sol, current_op, newls))
    end
    I0 = get_current(SciMLBase.NullParameters())
    stack(map(key -> (get_current(perturbations[[key]]) .- I0) / dμ, keys(perturbations)))
end