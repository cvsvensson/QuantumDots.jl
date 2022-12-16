
abstract type AbstractLead end
struct NormalLead{Op} <: AbstractLead
    temperature::Float64
    chemical_potential::Float64
    jump_in::Op
    jump_out::Op
    NormalLead(T,μ,jin::Op,jout::Op) where Op = new{Op}(T,μ,jin,jout)
end
struct OpenSystem{H,Ls}
    hamiltonian::H
    leads::Ls
end
struct DiagonalizedHamiltonian{Vals,Vecs}
    eigenvalues::Vals
    eigenvectors::Vecs
end
#kron(B,A)*vec(rho) ∼ A*rho*B'

# superjump(L) = kron(L,L) - 1/2*(kron(one(L),L'*L) + kron(L'*L,one(L)))
dissipator(L) = L⊗L - 1/2*kronsum(L'*L, L'*L)
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
diagonalize_hamiltonian(system::OpenSystem) = OpenSystem(DiagonalizedHamiltonian(eigen(Matrix(system.hamiltonian))...),leads(system))
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
    newmult! = _lindblad_with_normalizer(lindblad,n)#(out,in) = (mul!((@view out[2:end]),lindblad,in); out[1] = trnorm(in,n); out)
    #newmult(in) = pushfirst!(lindblad*in,trnorm(in))
    #newmult2(in) = pushfirst!(lindblad*in[2:end],trnorm(in[2:end]))
    #newmultadj(in) = lindblad'*(@view in[2:end])  + in[1]*idvec
    newmultadj! = _lindblad_with_normalizer_adj(lindblad,idvec)#(out,in) = (mul!(out,lindblad',(@view in[2:end]));  out .+= in[1]*idvec; out)
    #vin = rand(ComplexF64,n^2)
    #vout = similar(vin,n^2+1)
    #newmult!(vout,vin)
    #vout2 = newmult(vin)
    #println(norm(vout-vout2))
    #lm = LinearMap{ComplexF64}(newmult,newmultadj,n^2+1,n^2)
    lm! = LinearMap{ComplexF64}(newmult!,newmultadj!,n^2+1,n^2)
    #lm2 = LinearMap{ComplexF64}(newmult2,newmultadj2,n^2+1)
    v = Vector(sparsevec([1],ComplexF64[1.0],n^2+1))
    solver.x .= idvec ./ n
    sol = solve!(solver, lm!, v)
    # (xlsmr, stats!) = lsmr(lm!, v)
    #(xlsmr!, stats!) = lsmr(lm!, v)
    # display(stats)
    # display(stats!)
    # xlsmr
    sol.x
end
#(L::LinearMaps.LinearMap)(out,in::AbstractVector,p,t) = mul!(out,L,in)

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