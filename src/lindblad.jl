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

struct DiagonalizedHamiltonian{Vals,Vecs}
    eigenvalues::Vals
    eigenvectors::Vecs
end
Base.eltype(::DiagonalizedHamiltonian{Vals,Vecs}) where {Vals, Vecs} = promote_type(eltype(Vals),eltype(Vecs))

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

Base.show(io::IO, ::MIME"text/plain", system::OpenSystem) = show(io,system)
Base.show(io::IO, system::OpenSystem{H,Ls}) where {H,Ls} = print(io, "OpenSystem:\nHamiltonian: ",repr(system.hamiltonian),"\nleads: ", repr(system.leads))

Base.show(io::IO, ::MIME"text/plain", system::LindbladSystem) = show(io,system)
Base.show(io::IO, system::LindbladSystem) = print(io, "LindbladSystem:","\nOpenSystem",repr(system.system), 
    "\nUnitary: ", reprlindblad(system.unitary),"\ndissipators: ", reprdissipators(system.dissipators),
    "\nlindblad: ", reprlindblad(system.lindblad),
    "\nvectorizer: ", typeof(system.vectorizer))

reprlindblad(lm::LM) where {LM<:LinearMap} = "LinearMap{$(eltype(lm))}"
reprlindblad(m::AbstractMatrix) = typeof(m)
reprdissipators(x) = string(typeof(x), ", Labels: ", map(x->x.label,x)) #Base.repr(x)

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
const DENSE_CUTOFF = 6
dissipator(L,krv::KhatriRaoVectorizer) = sum(krv.sizes) > DENSE_CUTOFF ? khatri_rao_lazy_dissipator(L,krv.sizes) : khatri_rao_dissipator(Matrix(L),krv.sizes)
commutator(A,krv::KhatriRaoVectorizer) = sum(krv.sizes) > DENSE_CUTOFF ? khatri_rao_lazy_commutator(A,krv.sizes) : khatri_rao_commutator(Matrix(A),krv.sizes)

function dissipator(L,kv::KronVectorizer)
    D = (conj(L)⊗L - 1/2*kronsum(transpose(L'*L), L'*L))
    return kv.size > DENSE_CUTOFF ? D : Matrix(D)
end
commutator(A,::KronVectorizer) = commutator(A)
commutator(A) = kron(one(A),A) - kron(transpose(A),one(A))
measure(rho, op, ls::LindbladSystem) = map(d -> measure_dissipator(rho, op, ls.vectorizer, d) , ls.dissipators)

function measure_dissipator(rho, op::AbstractMatrix, vectorizer, dissipator::NamedTuple{(:in, :out, :label),<:Any})
    results = map(dissipator_op -> measure(rho,op,vectorizer,dissipator_op), (;dissipator.in,dissipator.out))
    merge(results,(;total = sum(results), label=dissipator.label))
end
measure(rho, op::AbstractMatrix, ::KronVectorizer, dissipator) = dot(conj(vec(op)), dissipator*vec(rho))
measure(rho::BlockDiagonal, op::BlockDiagonal,::KhatriRaoVectorizer, dissipator) = dot(conj(vecdp(op)), dissipator*vecdp(rho))
# measure(rho, op, dissipator) = dot(vec(op),dissipator*rho)
# current(ρ,op,sj) = tr(op * reshape(sj*ρ,size(op)))
# commutator(T1,T2) = -T1⊗T2 + T2⊗T1
# commutator(A) = -transpose(A)⊗one(A) + one(A)⊗A

hamiltonian(system) = system.hamiltonian
eigenvalues(system::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvalues(hamiltonian(system))
eigenvectors(system::OpenSystem{<:DiagonalizedHamiltonian}) = eigenvectors(hamiltonian(system))
eigenvalues(hamiltonian::DiagonalizedHamiltonian) = hamiltonian.eigenvalues
eigenvectors(hamiltonian::DiagonalizedHamiltonian) = hamiltonian.eigenvectors


khatri_rao_lazy_commutator(A, blocksizes) = khatri_rao_lazy(one(A),A,blocksizes) - khatri_rao_lazy(transpose(A),one(A),blocksizes) #Lazy seems slighly faster
khatri_rao_commutator(A, blocksizes) = khatri_rao(one(A),A,blocksizes) - khatri_rao(transpose(A),one(A),blocksizes) #Lazy seems slighly faster

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
function khatri_rao_dissipator(L,blocksizes)
    L2 = L'*L
    inds = sizestoinds(blocksizes)
    T = eltype(L)
    prodmaps = Matrix{T}[]
    summaps = Matrix{T}[]
    for ind1 in inds, ind2 in inds
        Lblock = L[ind1,ind2]
        leftprodmap = Lblock
        rightprodmap = conj(Lblock)
        push!(prodmaps, kron(rightprodmap,leftprodmap))
        if ind1==ind2
            L2block = L2[ind1,ind2]
            leftsummap = L2block
            rightsummap = transpose(L2block)
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
    maps = promote_type(typeof(L1),typeof(L2))[]
    for i in eachindex(blocksizes),j in eachindex(blocksizes)
        l1 = L1[inds[i],inds[j]]
        l2 = L2[inds[i],inds[j]]
        push!(maps, kron(l1,l2))
    end
    hvcat(length(inds),maps...)
end

khatri_rao(L1::Diagonal,L2::Diagonal) = kron(L1,L2)
khatri_rao(L1::BlockDiagonal,L2::BlockDiagonal) = cat([khatri_rao(B1,B2) for (B1,B2) in zip(blocks(L1),blocks(L2))]...; dims=(1,2))
function khatri_rao(L1::BlockDiagonal,L2::BlockDiagonal,bz) 
    if bz == first.(blocksizes(L1)) == first.(blocksizes(L2)) == last.(blocksizes(L1)) == last.(blocksizes(L2))
        return khatri_rao(L1,L2)
    else
        return khatri_rao(cat(L1.blocks...;dims=(1,2)),cat(L2.blocks...;dims=(1,2)),bz)
    end
end
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
    return NormalLead(T, μ, newjumpin, newjumpout, lead.label)
end
ratetransform(op,commutator_hamiltonian,T,μ) = reshape(sqrt(fermidirac(commutator_hamiltonian,T,μ))*vec(op),size(op))


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

_lindblad_with_normalizer(lindblad,kv::AbstractVectorizer) = (out,in) -> _lindblad_with_normalizer(out,in,lindblad,kv)
_lindblad_with_normalizer_adj(lindblad ,kv::AbstractVectorizer) = (out,in) -> _lindblad_with_normalizer_adj(out,in,lindblad,kv)

_lindblad_with_normalizer(out,in,lindblad,kv::KronVectorizer) = (mul!((@view out[1:end-1]),lindblad,in); out[end] = trnorm(in,kv.size); return out)
_lindblad_with_normalizer_adj(out,in, lindblad ,kv::KronVectorizer) = (mul!(out,lindblad',(@view in[1:end-1]));  out .+= in[end]*kv.idvec; return out)
_lindblad_with_normalizer(out,in,lindblad, krv::KhatriRaoVectorizer) = (mul!((@view out[1:end-1]),lindblad,in); out[end] = dot(krv.idvec, in); return out)
_lindblad_with_normalizer_adj(out,in,lindblad, krv::KhatriRaoVectorizer) = (mul!(out,lindblad',(@view in[1:end-1]));  out .+= in[end]*krv.idvec; return out)

Base.reshape(rho, vectorizer::KronVectorizer) = reshape(rho, vectorizer.size, vectorizer.size)
Base.reshape(rho, vectorizer::KhatriRaoVectorizer) = BlockDiagonal(map((size,inds)->reshape(rho[inds],size, size), vectorizer.sizes, sizestoinds(vectorizer.sizes .^2)))
stationary_state(lindbladsystem; solver = solver(lindbladsystem), kwargs...) = stationary_state(lindbladsystem, solver; kwargs...)
solver(ls::LindbladSystem) = LsmrSolver(size(ls.lindblad,1)+1, size(ls.lindblad,1), Vector{ComplexF64})


function stationary_state(lindbladsystem, solver; kwargs...)
    lindblad = lindbladsystem.lindblad
    vectorizer = lindbladsystem.vectorizer
    newmult! = QuantumDots._lindblad_with_normalizer(lindblad,vectorizer)
    newmultadj! = QuantumDots._lindblad_with_normalizer_adj(lindblad,vectorizer)
    n = size(lindblad,2)
    lm! = QuantumDots.LinearMap{ComplexF64}(newmult!,newmultadj!,n+1,n)
    x = zeros(eltype(lm!),n)
    push!(x,one(eltype(lm!)))
    solver.x .= vectorizer.idvec ./ sqrt(n)
    sol = Krylov.solve!(solver, lm!, x; kwargs...)
    reshape(sol.x, vectorizer), sol
end

function prepare_lindblad(system::OpenSystem, measurements; kwargs...)
    diagonalsystem = diagonalize(system; kwargs...)
    prepare_lindblad(diagonalsystem, measurements; kwargs...)
end
function prepare_lindblad(diagonalsystem::OpenSystem{<:DiagonalizedHamiltonian}, measurements; kwargs...)
    transformedsystem = ratetransform(diagonalsystem)
    vectorizer = default_vectorizer(diagonalsystem.hamiltonian)
    dissipators = map(lead->dissipator_from_transformed_lead(lead, vectorizer), transformedsystem.leads)
    unitary = -1im*commutator(eigenvalues(transformedsystem), vectorizer)
    lindblad = unitary + sum(d->d.in + d.out , dissipators)
    lindbladsystem = LindbladSystem(transformedsystem,unitary,dissipators,lindblad,vectorizer)
    transformedmeasureops = map(op->changebasis(op,lindbladsystem), measurements)
    return lindbladsystem, transformedmeasureops
end
changebasis(op,ls::LindbladSystem) = ls.system.hamiltonian.eigenvectors' * op * ls.system.hamiltonian.eigenvectors

function dissipator_from_transformed_lead(lead::NormalLead, vectorizer::AbstractVectorizer)
    opin = dissipator(lead.jump_in, vectorizer)
    opout = dissipator(lead.jump_out, vectorizer)
    (;in = opin, out = opout, label = lead.label)
end