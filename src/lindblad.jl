
abstract type AbstractLead end
struct NormalLead{Opin,Opout} <: AbstractLead
    temperature::Float64
    chemical_potential::Float64
    jump_in::Opin
    jump_out::Opout
    NormalLead(T,μ,jin::O1,jout::O2) where {O1,O2} = new{O1,O2}(T,μ,jin,jout)
end
struct OpenSystem{H,Ls}
    hamiltonian::H
    leads::Ls
end
struct DiagonalizedHamiltonian{Vals,Vecs}
    eigenvalues::Vals
    eigenvectors::Vecs
end
#kron(B,A)*vec(rho) ∼ A*rho*B' ∼ 
# A*rho*B = transpose(B)⊗A = kron(transpose(B),A)
# A*rho*transpose(B) = B⊗A = kron(B,A)

# superjump(L) = kron(L,L) - 1/2*(kron(one(L),L'*L) + kron(L'*L,one(L)))
dissipator(L) = (conj(L)⊗L - 1/2*kronsum(transpose(L'*L), L'*L))
current(ρ,op,sj) = tr(op * reshape(sj*ρ,size(op)))
# commutator(T1,T2) = -T1⊗T2 + T2⊗T1
commutator(A) = kron(one(A),A) - kron(A,one(A))
function transform_jump_op(H::Diagonal,L,T,μ)
    comm = commutator(H)
    reshape(sqrt(fermidirac(comm,T,μ))*vec(L),size(L))
end
hamiltonian(system) = system.hamiltonian
eigenvalues(system::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvalues(hamiltonian(system))
eigenvectors(system::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvectors(hamiltonian(system))
eigenvalues(hamiltonian::DiagonalizedHamiltonian) = hamiltonian.eigenvalues
eigenvectors(hamiltonian::DiagonalizedHamiltonian) = hamiltonian.eigenvectors

function ratetransform(system::OpenSystem{<:DiagonalizedHamiltonian})
    comm = commutator(Diagonal(eigenvalues(system)))
    newleads = [ratetransform(lead,comm) for lead in system.leads]
    return OpenSystem(hamiltonian(system), newleads)
end
chemical_potential(lead::NormalLead) = lead.chemical_potential
temperature(lead::NormalLead) = lead.temperature

function ratetransform(lead::NormalLead, commutator_hamiltonian)
    μ = chemical_potential(lead)
    T = temperature(lead)
    Lin = lead.jump_in
    Lout = lead.jump_out
    newjumpin = reshape(sqrt(fermidirac(commutator_hamiltonian,T,μ))*vec(Lin),size(Lin))
    newjumpout = reshape(sqrt(fermidirac(commutator_hamiltonian,T,-μ))*vec(Lout),size(Lout))
    return NormalLead(T, μ, newjumpin, newjumpout)
end

diagonalize(S,lead::NormalLead) = NormalLead(lead.temperature, lead.chemical_potential, S'*lead.jump_in*S,  S'*lead.jump_out*S)
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
diagonalize(system::OpenSystem) = diagonalize_leads(diagonalize_hamiltonian(system))

function LinearAlgebra.eigen((Heven,Hodd); kwargs...)
    oeigvals, oeigvecs = eigen(Heven; kwargs...)
	eeigvals, eeigvecs = eigen(Hodd; kwargs...)
	eigvals = vcat(oeigvals,eeigvals)
	S = cat(sparse(oeigvecs),sparse(eeigvecs); dims=[1,2])
    D = Diagonal(eigvals)
    return D, S
end
kronblocksizes(A,B) = map(Ab->size(Ab) .* size(B),A.blocks)

function LinearAlgebra.kron(A::BlockDiagonal{TA,VA}, B::BlockDiagonal{TB,VB}) where {TA,TB,VA,VB}
    VC = promote_type(VA,VB)
    TC = promote_type(TA,TB)
    C::BlockDiagonal{TC,VC} = BlockDiagonal(map(Ab-> similar(Ab,size(Ab) .* size(B)), A.blocks))
    # C::BlockDiagonal{TC,VC} = BlockDiagonal(map(Ab-> VC(zeros(size(Ab) .* size(B))), A.blocks))
    kron!(C,A,B)
end
function LinearAlgebra.kron(A::BlockDiagonal{TA,VA}, B::BlockDiagonal{TB,VB}) where {TA,TB,VA<:Diagonal,VB<:Diagonal}
    VC = promote_type(VA,VB)
    TC = promote_type(TA,TB)
    C::BlockDiagonal{TC,VC} = BlockDiagonal(map(Ab-> Diagonal{TC}(undef, size(Ab,1) .* size(B,1)), A.blocks))
    kron!(C,A,B)
end
Base.convert(::Type{D},bd::BlockDiagonal{<:Any,D}) where D<:Diagonal = Diagonal(bd)
function LinearAlgebra.kron!(C::BlockDiagonal, A::BlockDiagonal, B::BlockDiagonal{<:Any,V}) where V
    bmat = convert(V,B)
    for (Cb,Ab) in zip(C.blocks,A.blocks)
        kron!(Cb, Ab, bmat)
    end
    return C
end

LinearAlgebra.exp(D::BlockDiagonal) = BlockDiagonal(map(LinearAlgebra.exp, D.blocks))
LinearAlgebra.sqrt(D::BlockDiagonal) = BlockDiagonal([promote(map(LinearAlgebra.sqrt, D.blocks)...)...])

for f in (:cis, :log,
    :cos, :sin, :tan, :csc, :sec, :cot,
    :cosh, :sinh, :tanh, :csch, :sech, :coth,
    :acos, :asin, :atan, :acsc, :asec, :acot,
    :acosh, :asinh, :atanh, :acsch, :asech, :acoth,
    :one)
@eval Base.$f(D::BlockDiagonal) = BlockDiagonal(map(Base.$f, D.blocks))
end

# LinearAlgebra.sqrt(A::BlockDiagonal) = BlockDiagonal([promote(map(sqrt, A.blocks)...)...])

fermidirac(E,T,μ) = (I + exp(E/T)exp(-μ/T))^(-1)

leads(system::OpenSystem) = system.leads
jumpins(system::OpenSystem) = [lead.jump_in for lead in leads(system)]
jumpouts(system::OpenSystem) = [lead.jump_out for lead in leads(system)]
jumpops(system::OpenSystem) = vcat(jumpins(system),jumpouts(system))


lindbladian(system::OpenSystem{<:DiagonalizedHamiltonian}) = lindbladian(Diagonal(eigenvalues(system)), jumpops(system))
function lindbladian(hamiltonian,Ls)
    id = one(hamiltonian)
    -1im*commutator(hamiltonian) + sum(Ls, init = 0*kron(id,id))
end

trnorm(rho,n) = tr(reshape(rho,n,n))
_lindblad_with_normalizer(lindblad,n) = (out,in) -> (mul!((@view out[2:end]),lindblad,in); out[1] = trnorm(in,n);)
_lindblad_with_normalizer_adj(lindblad,idvec) = (out,in) -> (mul!(out,lindblad',(@view in[2:end]));  out .+= in[1]*idvec;)
function stationary_state(lindblad; solver = LsmrSolver(size(lindblad,1)+1,size(lindblad,1),Vector{ComplexF64}))
    n = Int(sqrt(size(lindblad,1)))
    idvec = vec(Matrix(I,n,n))
    newmult! = _lindblad_with_normalizer(lindblad,n)
    newmultadj! = _lindblad_with_normalizer_adj(lindblad,idvec)
    lm! = LinearMap{ComplexF64}(newmult!,newmultadj!,n^2+1,n^2)
    v = Vector(sparsevec([1],ComplexF64[1.0],n^2+1))
    solver.x .= idvec ./ n
    sol = solve!(solver, lm!, v)
    sol.x
end

function conductance(system::OpenSystem, measureops; kwargs...)
    diagonalsystem = diagonalize(system)
    transformedsystem = ratetransform(diagonalsystem)
    superjumpins = dissipator.(jumpins(transformedsystem))
    superjumpouts = dissipator.(jumpouts(transformedsystem))
    superlind = lindbladian(Diagonal(eigenvalues(transformedsystem)), vcat(superjumpins,superjumpouts))
    ρ = stationary_state(superlind; kwargs...)
    S = eigenvectors(transformedsystem)
    transformedmeasureops = map(op->S'*(op)*S,measureops)
    [hcat(map(sj->current(ρ, op, sj), superjumpins), map(sj->current(ρ, op, sj), superjumpouts)) for op in transformedmeasureops]
end