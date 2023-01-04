
abstract type AbstractLead end
struct NormalLead{Opin,Opout} <: AbstractLead
    temperature::Float64
    chemical_potential::Float64
    jump_in::Opin
    jump_out::Opout
    NormalLead(T,μ,jin::O1,jout::O2) where {O1,O2} = new{O1,O2}(T,μ,jin,jout)
end
NormalLead(T,μ; in = jin, out = jout) = NormalLead(T,μ,in,out)


abstract type AbstractOpenSystem end
struct OpenSystem{H,Ls} <: AbstractOpenSystem
    hamiltonian::H
    leads::Ls
end
struct LindbladSystem{O,U,Ds,L,V} <: AbstractOpenSystem
    system::O
    unitary::U
    dissipators::Ds
    lindblad::L
    vectorizer::V
end

struct DiagonalizedHamiltonian{Vals,Vecs}
    eigenvalues::Vals
    eigenvectors::Vecs
end
Base.eltype(::DiagonalizedHamiltonian{Vals,Vecs}) where {Vals, Vecs} = promote_type(eltype(Vals),eltype(Vecs))

abstract type AbstractVectorizer end
struct KronVectorizer{T} <: AbstractVectorizer
    size::Int
    idvec::Vector{T}
end
struct KhatriRaoVectorizer{T} <: AbstractVectorizer
    sizes::Vector{Int}
    idvec::Vector{T}
end
function KronVectorizer(ham::DiagonalizedHamiltonian)
    n = size(ham.eigenvalues,1)
    KronVectorizer{eltype(ham)}(n, vec(Matrix(I,n,n)))
end
function KhatriRaoVectorizer(ham::DiagonalizedHamiltonian)
    sizes = first.(blocksizes(ham.eigenvalues))
    blockid = BlockDiagonal([Matrix{eltype(ham)}(I,size,size) for size in sizes])
    KhatriRaoVectorizer{eltype(ham)}(sizes, vecdp(blockid))
end
default_vectorizer(ham::DiagonalizedHamiltonian{<:BlockDiagonal}) = KhatriRaoVectorizer(ham)
default_vectorizer(ham::DiagonalizedHamiltonian) = KronVectorizer(ham)

#kron(B,A)*vec(rho) ∼ A*rho*B' ∼ 
# A*rho*B = transpose(B)⊗A = kron(transpose(B),A)
# A*rho*transpose(B) = B⊗A = kron(B,A)
# superjump(L) = kron(L,L) - 1/2*(kron(one(L),L'*L) + kron(L'*L,one(L)))
dissipator(L,krv::KhatriRaoVectorizer) = khatri_rao_lazy_dissipator(L,krv.sizes)
commutator(A,krv::KhatriRaoVectorizer) = khatri_rao_commutator(A,krv.sizes)
dissipator(L,::KronVectorizer) = (conj(L)⊗L - 1/2*kronsum(transpose(L'*L), L'*L))
commutator(A,::KronVectorizer) = commutator(A)
commutator(A) = kron(one(A),A) - kron(transpose(A),one(A))
measure(rho, op::AbstractMatrix, ls::LindbladSystem) = measure(rho,op,ls.vectorizer, ls.dissipators)
measure(rho, op::AbstractMatrix, ::KronVectorizer, dissipators) = map(dissipator->dot(conj(vec(op)), dissipator*vec(rho)), dissipators)
measure(rho::BlockDiagonal, op::BlockDiagonal,::KhatriRaoVectorizer, dissipators) = map(dissipator->dot(conj(vecdp(op)), dissipator*vecdp(rho)), dissipators)
# measure(rho, op, dissipator) = dot(vec(op),dissipator*rho)
# current(ρ,op,sj) = tr(op * reshape(sj*ρ,size(op)))
# commutator(T1,T2) = -T1⊗T2 + T2⊗T1
# commutator(A) = -transpose(A)⊗one(A) + one(A)⊗A

hamiltonian(system) = system.hamiltonian
eigenvalues(system::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvalues(hamiltonian(system))
eigenvectors(system::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvectors(hamiltonian(system))
eigenvalues(hamiltonian::DiagonalizedHamiltonian) = hamiltonian.eigenvalues
eigenvectors(hamiltonian::DiagonalizedHamiltonian) = hamiltonian.eigenvectors

# lindbladian(system::OpenSystem{<:DiagonalizedHamiltonian}) = lindbladian(eigenvalues(system), jumpops(system))
# function lindbladian(hamiltonian, dissipators)
#     #id = one(hamiltonian)
#     -1im*commutator(hamiltonian) + sum(dissipators)#, init = 0*kron(id,id))
# end

khatri_rao_commutator(A, blocksizes) = khatri_rao_lazy(one(A),A,blocksizes) - khatri_rao_lazy(transpose(A),one(A),blocksizes) #Lazy seems slighly faster

function khatri_rao_lazy_dissipator(L,blocksizes)
    L2 = L'*L
    inds = sizestoinds(blocksizes)
    T = eltype(L)
    prodmaps = LinearMap{T}[]
    summaps = LinearMap{T}[]
    for ind1 in inds, ind2 in inds
        Lblock = L[ind1,ind2]
        leftprodmap = LinearMap{T}(Lblock)
        rightprodmap = LinearMap{T}(conj(Lblock))
        push!(prodmaps, kron(rightprodmap,leftprodmap))
        if ind1==ind2
            L2block = L2[ind1,ind2]
            leftsummap = LinearMap{T}(L2block)
            rightsummap = LinearMap{T}(transpose(L2block))
            push!(summaps, kronsum(rightsummap,leftsummap))
        end
    end
    hvcat(length(inds),prodmaps...) - 1/2*cat(summaps...; dims=(1,2))
end

function khatri_rao_lazy(L1,L2,blocksizes)
    inds = sizestoinds(blocksizes)
    T = promote_type(eltype(L1),eltype(L2))
    maps = LinearMap{T}[]
    for i in eachindex(blocksizes),j in eachindex(blocksizes)
        L1bij = L1[inds[i],inds[j]]
        L2bij = L2[inds[i],inds[j]]
        l1 = LinearMap{T}(L1bij)
        l2 = LinearMap{T}(L2bij)
        push!(maps, kron(l1,l2))
    end
    hvcat(length(inds),maps...)
end
function khatri_rao(L1,L2,blocksizes)
    inds = sizestoinds(blocksizes)
    T = promote_type(eltype(L1),eltype(L2))
    maps = []
    for i in eachindex(blocksizes),j in eachindex(blocksizes)
        l1 = L1[inds[i],inds[j]]
        l2 = L2[inds[i],inds[j]]
        push!(maps, kron(l1,l2))
    end
    hvcat(length(inds),maps...)
end
# function khatri_rao(L1::Diagonal,L2::Diagonal,blocksizes)
#     inds = sizestoinds(blocksizes)
#     T = promote_type(eltype(L1),eltype(L2))
#     l1 = parent(L1)
#     l2 = parent(L2)
#     diagonals = Vector{T}[]
#     for inds in inds#, j in eachindex(blocksizes)
#         push!(diagonals, diag(kron(Diagonal(l1[inds]),Diagonal(l2[inds]))))
#     end
#     Diagonal(reduce(vcat,diagonals))
# end
khatri_rao(L1::Diagonal,L2::Diagonal) = kron(L1,L2)
khatri_rao(L1::BlockDiagonal,L2::BlockDiagonal) = cat([khatri_rao(B1,B2) for (B1,B2) in zip(blocks(L1),blocks(L2))]...; dims=(1,2))
function khatri_rao(L1::BlockDiagonal,L2::BlockDiagonal,bz) 
    if bz == first.(blocksizes(L1)) == first.(blocksizes(L2)) == last.(blocksizes(L1)) == last.(blocksizes(L2))
        return khatri_rao(L1,L2)
    else
        return khatri_rao(sparse(L1),sparse(L2),bz)
    end
end

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
    newjumpin = ratetransform(lead.jump_in,commutator_hamiltonian,T,μ) #reshape(sqrt(fermidirac(commutator_hamiltonian,T,μ))*vec(Lin),size(Lin))
    newjumpout = ratetransform(lead.jump_out,commutator_hamiltonian,T,-μ) #reshape(sqrt(fermidirac(commutator_hamiltonian,T,-μ))*vec(Lout),size(Lout))
    return NormalLead(T, μ, newjumpin, newjumpout)
end
ratetransform(op,commutator_hamiltonian,T,μ) = reshape(sqrt(fermidirac(commutator_hamiltonian,T,μ))*vec(op),size(op))


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
jumpins(system::AbstractOpenSystem) = [lead.jump_in for lead in leads(system)]
jumpouts(system::AbstractOpenSystem) = [lead.jump_out for lead in leads(system)]
jumpops(system::AbstractOpenSystem) = hcat(jumpins(system),jumpouts(system))

trnorm(rho,n) = tr(reshape(rho,n,n))
#khatri_rao_trnorm(rho,blocksizes) =  [ for size in blocksizes]  #mapreduce(trnorm,+,rho,blocksizes)
vecdp(bd::BlockDiagonal) = mapreduce(vec, vcat, blocks(bd))

_lindblad_with_normalizer(lindblad,kv::KronVectorizer) = (out,in) -> (mul!((@view out[2:end]),lindblad,in); out[1] = trnorm(in,kv.size);)
_lindblad_with_normalizer_adj(lindblad ,kv::KronVectorizer) = (out,in) -> (mul!(out,lindblad',(@view in[2:end]));  out .+= in[1]*kv.idvec;)
_lindblad_with_normalizer(lindblad, krv::KhatriRaoVectorizer) = (out,in) -> (mul!((@view out[2:end]),lindblad,in); out[1] = dot(krv.idvec, in);)#khatri_rao_trnorm(in,krv.sizes);)
_lindblad_with_normalizer_adj(lindblad, krv::KhatriRaoVectorizer) = (out,in) -> (mul!(out,lindblad',(@view in[2:end]));  out .+= in[1]*krv.idvec;)

function stationary_state(lindbladsystem, solver)
    lindblad = lindbladsystem.lindblad
    vectorizer = lindbladsystem.vectorizer
    newmult! = _lindblad_with_normalizer(lindblad,vectorizer)
    newmultadj! = _lindblad_with_normalizer_adj(lindblad,vectorizer)
    n = size(lindblad,2)
    lm! = LinearMap{ComplexF64}(newmult!,newmultadj!,n+1,n)
    v = Vector(sparsevec([1],ComplexF64[1.0],n+1))
    solver.x .= vectorizer.idvec ./ n
    sol = solve!(solver, lm!, v)
    Matrix(sol.x, vectorizer)
end

# Base.Vector(mat, vectorizer::KronVectorizer) = vec(mat)
# Base.Vector(mat, vectorizer::KhatriRaoVectorizer) = BlockDiagonal(map(inds->reshape(rho[inds],length(inds),length(inds)), sizestoinds(vectorizer.sizes)))
Base.Matrix(rho::Vector, vectorizer::KronVectorizer) = reshape(rho, vectorizer.size,vectorizer.size)
Base.Matrix(rho::Vector, vectorizer::KhatriRaoVectorizer) = BlockDiagonal(map((size,inds)->reshape(rho[inds],size, size), vectorizer.sizes, sizestoinds(vectorizer.sizes .^2)))
stationary_state(lindbladsystem; solver = solver(lindbladsystem)) = stationary_state(lindbladsystem, solver)
solver(ls::LindbladSystem) = LsmrSolver(size(ls.lindblad,1)+1, size(ls.lindblad,1), Vector{ComplexF64})
#     n = Int(sqrt(size(lindblad,1)))
#     idvec = vec(Matrix(I,n,n))
#     newmult! = _lindblad_with_normalizer(lindblad,n)
#     newmultadj! = _lindblad_with_normalizer_adj(lindblad,idvec)
#     lm! = LinearMap{ComplexF64}(newmult!,newmultadj!,n^2+1,n^2)
#     v = Vector(sparsevec([1],ComplexF64[1.0],n^2+1))
#     solver.x .= idvec ./ n
#     sol = solve!(solver, lm!, v)
#     sol.x
# end

function prepare_lindblad(system, measurements)
    diagonalsystem = diagonalize(system)
    transformedsystem = ratetransform(diagonalsystem)
    vectorizer = default_vectorizer(diagonalsystem.hamiltonian)
    superjumpins = map(op->dissipator(op,vectorizer), jumpins(transformedsystem))
    superjumpouts = map(op->dissipator(op,vectorizer), jumpouts(transformedsystem))
    unitary = -1im*commutator(eigenvalues(transformedsystem), vectorizer)
    dissipators = hcat(superjumpins,superjumpouts)
    lindblad = unitary + sum(dissipators)
    lindbladsystem = LindbladSystem(transformedsystem,unitary,dissipators,lindblad,vectorizer)
    
    transformedmeasureops = map(op->changebasis(op,lindbladsystem), measurements)
    return lindbladsystem, transformedmeasureops
end
changebasis(op,ls::LindbladSystem) = ls.system.hamiltonian.eigenvectors' * op * ls.system.hamiltonian.eigenvectors

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