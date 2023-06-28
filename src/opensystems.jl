abstract type AbstractLead end
struct NormalLead{T,Opin,Opout,L} <: AbstractLead
    temperature::T
    chemical_potential::T
    jump_in::Opin
    jump_out::Opout
    label::L
end
NormalLead(temp,μ,jin,jout; label = missing) = NormalLead(temp,μ,jin,jout, label)
NormalLead(temp,μ,jin; label = missing) = NormalLead(temp,μ,jin,jin', label)
NormalLead(T,μ; in, out = in', label = missing) = NormalLead(T,μ,in,out, label)

Base.show(io::IO, ::MIME"text/plain", lead::NormalLead{T,Opin,Opout,N}) where {T,Opin,Opout,N} = print(io, "NormalLead{$T,$Opin,$Opout,$N}(Label=",lead.label, ", T=",lead.temperature,", μ=", lead.chemical_potential,")")
Base.show(io::IO, lead::NormalLead{T,Opin,Opout,N}) where {T,Opin,Opout,N} = print(io, "Lead(",lead.label, ", T=", round(lead.temperature,digits=4),", μ=", round(lead.chemical_potential, digits=4),")")

chemical_potential(lead::NormalLead) = lead.chemical_potential
temperature(lead::NormalLead) = lead.temperature

abstract type AbstractOpenSystem end
struct OpenSystem{H,Ls} <: AbstractOpenSystem
    hamiltonian::H
    leads::Ls
end
eigenvaluevector(H::OpenSystem{<:DiagonalizedHamiltonian}) = diag(eigenvalues(H))
hamiltonian(system::OpenSystem) = system.hamiltonian
eigenvalues(system::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvalues(hamiltonian(system))
eigenvectors(system::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvectors(hamiltonian(system))
eigenvalues(hamiltonian::DiagonalizedHamiltonian) = hamiltonian.eigenvalues
eigenvectors(hamiltonian::DiagonalizedHamiltonian) = hamiltonian.eigenvectors


Base.show(io::IO, ::MIME"text/plain", system::OpenSystem) = show(io,system)
Base.show(io::IO, system::OpenSystem{H,Ls}) where {H,Ls} = print(io, "OpenSystem:\nHamiltonian: ",repr(system.hamiltonian),"\nleads: ", repr(system.leads))


abstract type AbstractOpenSolver end

struct OpenSystemLinearProblem{S,LP} <: SciMLBase.AbstractLinearProblem
    system::S
    linearproblem::LP
end
struct OpenSystemODEProblem{S,OP} <: SciMLBase.AbstractODEProblem
    system::S
    odeproblem::OP
end

function LinearSolve.LinearProblem(system::AbstractOpenSystem, args...; kwargs...)
    OpenSystemLinearProblem(system, LinearProblem(LinearOperator(system; kwargs...), args...; kwargs...))
end
function OrdinaryDiffEq.ODEProblem(system::AbstractOpenSystem, args...; kwargs...)
    OpenSystemODEProblem(system, ODEProblem(LinearOperator(system; kwargs...), args...; kwargs...))
end

LinearOperator(mat::AbstractMatrix; kwargs...) = MatrixOperator(mat; kwargs...)
LinearOperator(func::Function; kwargs...) = FunctionOperator(func; islinear = true, kwargs...)



function ratetransform(system::OpenSystem{<:DiagonalizedHamiltonian})
    comm = commutator(Diagonal(eigenvalues(system)))
    newleads = [ratetransform(lead,comm) for lead in system.leads]
    return OpenSystem(hamiltonian(system), newleads)
end


function ratetransform(lead::NormalLead, commutator_hamiltonian)
    μ = chemical_potential(lead)
    T = temperature(lead)
    newjumpin = ratetransform(lead.jump_in,commutator_hamiltonian,T,μ) #reshape(sqrt(fermidirac(commutator_hamiltonian,T,μ))*vec(Lin),size(Lin))
    newjumpout = ratetransform(lead.jump_out,commutator_hamiltonian,T,-μ) #reshape(sqrt(fermidirac(commutator_hamiltonian,T,-μ))*vec(Lout),size(Lout))
    return NormalLead(T, μ, newjumpin, newjumpout, lead.label)
end
ratetransform(op,commutator_hamiltonian::Diagonal,T,μ) = reshape(sqrt(fermidirac(commutator_hamiltonian,T,μ))*vec(op),size(op))


diagonalize(S,lead::NormalLead) = NormalLead(lead.temperature, lead.chemical_potential, S'*lead.jump_in*S,  S'*lead.jump_out*S,lead.label)
diagonalize_hamiltonian(system::OpenSystem) = OpenSystem(diagonalize(hamiltonian(system)), leads(system))

function diagonalize(m::AbstractMatrix)
    vals, vecs = eigen(m)
    DiagonalizedHamiltonian(Diagonal(vals), vecs)
end
diagonalize(m::SparseMatrixCSC) = diagonalize(Matrix(m))
function diagonalize(m::BlockDiagonal)
    vals,vecs = BlockDiagonals.eigen_blockwise(m)
    blockinds = sizestoinds(map(first,blocksizes(vecs)))
    bdvals = BlockDiagonal(map(inds -> Diagonal(vals[inds]), blockinds))
    DiagonalizedHamiltonian(bdvals,vecs)
end
diagonalize(m::BlockDiagonal{<:Any,<:SparseMatrixCSC}) = diagonalize(BlockDiagonal(Matrix.(m.blocks)))
diagonalize(m::BlockDiagonal{<:Any,<:Hermitian{<:Any,<:SparseMatrixCSC}}) = diagonalize(BlockDiagonal(Hermitian.(Matrix.(m.blocks))))

diagonalize_leads(system::OpenSystem{<:DiagonalizedHamiltonian}) = OpenSystem(hamiltonian(system), [diagonalize(eigenvectors(system), lead) for lead in leads(system)])
function diagonalize(system::OpenSystem; dE = 0.0) 
    diagonal_system = diagonalize_hamiltonian(system)
    if dE > 0
        diagonal_system = remove_high_energy_states(dE, diagonal_system)
    end
    diagonalize_leads(diagonal_system)
end

fermidirac(E,T,μ) = (I + exp(E/T)exp(-μ/T))^(-1)

leads(system::OpenSystem) = system.leads

trnorm(rho,n) = tr(reshape(rho,n,n))
vecdp(bd::BlockDiagonal) = mapreduce(vec, vcat, blocks(bd))


remove_high_energy_states(dE,system::OpenSystem) = OpenSystem(remove_high_energy_states(dE,hamiltonian(system)),leads(system))
function remove_high_energy_states(ΔE,ham::DiagonalizedHamiltonian{<:BlockDiagonal,<:BlockDiagonal})
    vals = eigenvalues(ham)
    vecs = eigenvectors(ham)
    E0 = minimum(diag(vals))
    Is = map(vals->findall(<(ΔE+E0),diag(vals)), blocks(vals))
    newblocks = map((block,I)-> block[:,I],blocks(vecs),Is)
    newvals = map((vals,I)-> Diagonal(diag(vals)[I]), blocks(vals), Is)
    DiagonalizedHamiltonian(BlockDiagonal(newvals), BlockDiagonal(newblocks))
end
function remove_high_energy_states(ΔE,ham::DiagonalizedHamiltonian)
    vals = eigenvalues(ham)
    vecs = eigenvectors(ham)
    E0 = minimum(diag(vals))
    I = findall(<(ΔE+E0),diag(vals))
    newvecs = vecs[:,I]
    newvals = Diagonal(diag(vals)[I])
    DiagonalizedHamiltonian(newvals, newvecs)
end
