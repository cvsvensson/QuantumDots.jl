
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
struct DiagonalizedOpenSystem{D,S,Ls}
    eigenvalues::D
    eigenvectors::S
    leads::Ls
end
#kron(B,A)*vec(rho) ∼ A*rho*B'

# superjump(L) = kron(L,L) - 1/2*(kron(one(L),L'*L) + kron(L'*L,one(L)))
function superjump!((c1,c2,c3), L)
    kron!(c1,L,L)
    kron!(c2,one(L),L'*L)
    kron!(c3,L'*L,one(L))
    c1 - 1/2*(c2+c3) 
end
superjump(L) = L⊗L - 1/2*kronsum((L'*L),L'*L)
current(ρ,op,sj) = tr(op * reshape(sj*ρ,size(op)))
# commutator(T1,T2) = -T1⊗T2 + T2⊗T1
commutator(A) = kron(one(A),A) - kron(A,one(A))
function commutator!((cout,c1,c2),A)
    kron!(c1,one(A),A)
    kron!(c2,A,one(A))
    c1 - c2
end
function transform_jump_op(H::Diagonal,L,T,μ)
    comm = commutator(H)
    reshape(sqrt(fermidirac(comm,T,μ))*vec(L),size(L))
end
function ratetransform(system::DiagonalizedOpenSystem)
    comm = commutator(Diagonal(system.eigenvalues))
    newleads = [ratetransform(lead,comm) for lead in system.leads]
    return DiagonalizedOpenSystem(system.eigenvalues,system.eigenvectors,newleads)
end
function ratetransform(lead::NormalLead, commutator)
    μ = lead.chemical_potential
    T = lead.temperature
    Lin = lead.jump_in
    Lout = lead.jump_out
    newjumpin = reshape(sqrt(fermidirac(commutator,T,μ))*vec(Lin),size(Lin))
    newjumpout = reshape(sqrt(fermidirac(commutator,T,-μ))*vec(Lout),size(Lout))
    return NormalLead(T, μ, newjumpin, newjumpout)
end

function diagonalize(S,lead::NormalLead)
    newjump_in = S'*lead.jump_in*S
    newjump_out = S'*lead.jump_out*S
    NormalLead(lead.temperature, lead.chemical_potential, newjump_in, newjump_out)
end
function diagonalize(system::OpenSystem)
    vals, vecs = eigen(Matrix(system.hamiltonian))
    newleads = [diagonalize(vecs, lead) for lead in system.leads]
    DiagonalizedOpenSystem(vals,vecs,newleads)
end

function LinearAlgebra.eigen((Heven,Hodd); kwargs...)
    oeigvals, oeigvecs = eigen(Heven; kwargs...)
	eeigvals, eeigvecs = eigen(Hodd; kwargs...)
	eigvals = vcat(oeigvals,eeigvals)
	S = cat(sparse(oeigvecs),sparse(eeigvecs); dims=[1,2])
    D = Diagonal(eigvals)
    return D, S
end

fermidirac(E::Diagonal,T,μ) = (I + exp(E/T)exp(-μ/T))^(-1)

leads(system::Union{DiagonalizedOpenSystem,OpenSystem}) = system.leads
jumpins(system::Union{DiagonalizedOpenSystem,OpenSystem}) = [lead.jump_in for lead in leads(system)]
jumpouts(system::Union{DiagonalizedOpenSystem,OpenSystem}) = [lead.jump_out for lead in leads(system)]
jumpops(system::Union{DiagonalizedOpenSystem,OpenSystem}) = vcat(jumpins(system),jumpouts(system))

function lindbladian(system::DiagonalizedOpenSystem)
    # id = one(system.eigenvalues)
    # Ls = jumpops(system)
    lindbladian(Diagonal(system.eigenvalues), jumpops(system))
    # -1im*commutator(system.eigenvalues) + sum(superjump, Ls, init = 0*kron(id,id))
end
function lindbladian(hamiltonian,Ls)
    id = one(hamiltonian)
    -1im*commutator(hamiltonian) + sum(Ls, init = 0*kron(id,id))
end

###
function stationary_state(lindbladian; kwargs...)
    eigsolve(lindbladian,size(lindbladian,1),1,EigSorter(abs; rev=false); kwargs...)
end
using KrylovKit
using Krylov

function normalizedlinsolve(lindblad)
    n = Int(sqrt(size(lindblad,1)))
    trnorm(rho) = tr(reshape(rho,n,n))
    #v0 = rand(ComplexF64,n^2)
    # v2 = deepcopy(v)
    # mul!(v2,lindblad,v)
    #newmult!(out,in) = (mul!(0*out[2:end],lindblad,in); out[1] = trnorm(in))
    newmult(in) = pushfirst!(lindblad*in,trnorm(in))
    #newmult2(in) = pushfirst!(lindblad*in[2:end],trnorm(in[2:end]))
    newmultadj(in) = lindblad'*in[2:end]  + in[1]*vec(Matrix(I,n,n))#+ Vector(vec(Diagonal(reshape(in[2:end],n,n))))
    #newmultadj2(in) = pushfirst!(lindblad'*in[2:end] + in[1]*vec(Matrix(I,n,n)),0.0*in[1])
    #newmult!2(out,in,u,p) = (mul!(out[2:end],lindblad,in); out[1] = trnorm(in))
    lm = LinearMap{ComplexF64}(newmult,newmultadj,n^2+1,n^2)
    #lm2 = LinearMap{ComplexF64}(newmult2,newmultadj2,n^2+1)
    #println("lv: ", lm*v0)
    v = Vector(sparsevec([1],ComplexF64[1.0],n^2+1))
    # prob = LinearProblem(lm, v)
    # display(prob)
    #println("sols")
    # m = Matrix(lm)
    # m2 = Matrix(lm')
    # println(norm(m'-m2))
    # println(norm(Matrix(lm2)'-Matrix(lm2')))
    #display(m')
    #display(m2)
    #@time sol = solve(LinearProblem(lm, v))
    # @time sol2 = solve(LinearProblem(lm2, v))
    # @time solk= solve(LinearProblem(lm, v), KrylovJL_LSMR())
    # @time solk2 = solve(LinearProblem(lm2, v), KrylovJL_LSMR())
    (xlsmr, stats) = lsmr(lm, v)
    # @time (xlsmr2, stats) = lsmr(lm2, v)
    # @time (xcls, stats) = cgls(lm, v)
    # @time (xcls2, stats) = cgls(lm2, v)
    # @time sole = eigsolve(lindblad,n^2,1, EigSorter(abs; rev=false))
#    println(norm(sol))
    # println(norm(sol2))
    # println(norm(solk))
    # println(norm(solk2))
    # println(norm(xlsmr))
    # println(norm(xlsmr2))
    # println(norm(xcls))
    # println(norm(xcls2))
    #println(norm(xcls-sol))
    # println("eigsolve: ", sole[1][1])
    # println("solve:", norm(sol/norm(sol) - sol2[2:end]/norm(sol2[2:end])))
    #@time sol2 = linsolve(lm2,v)
    #println(sol.u)
    xlsmr
end
#(L::LinearMaps.LinearMap)(out,in::AbstractVector,p,t) = mul!(out,L,in)

function conductance(system::OpenSystem, measureops; kwargs...)
    diagonalsystem = diagonalize(system)
    transformedsystem = ratetransform(diagonalsystem)
    #caches = ntuple(i->spzeros(length(transformedsystem.eigenvalues)^2,length(transformedsystem.eigenvalues)^2), 3)

    superjumpins = superjump.(jumpins(transformedsystem))
    superjumpouts = superjump.(jumpouts(transformedsystem))
    #superjumpins = map(op->superjump!(caches,op),jumpins(transformedsystem))
    #superjumpouts =  map(op->superjump!(caches,op),jumpouts(transformedsystem))
    superlind = lindbladian(Diagonal(transformedsystem.eigenvalues), vcat(superjumpins,superjumpouts))
    #@time lindvals,lindvecs = eigsolve(superlind, size(superlind,1), 1, EigSorter(abs; rev=false);kwargs...)
    #println(lindvals)
    lindvecs2 = normalizedlinsolve(superlind)
    S = transformedsystem.eigenvectors
    transformedmeasureops = map(op->S'*(op)*S,measureops)
    normalized_stationary_state = lindvecs2#first(lindvecs)/(tr(reshape(first(lindvecs),size(S))))
    [[map(sj->current(normalized_stationary_state, op, sj), superjumpins) ;; map(sj->current(normalized_stationary_state, op, sj), superjumpouts)] for op in transformedmeasureops]
end

function conductance2(p0,μL,μR,leftleadops,rightleadops,measureop,TL,TR)
	Hs = densehamiltonian![N](p0)
	oeigvals, oeigvecs = eigen!(Hs[1])
	eeigvals, eeigvecs = eigen!(Hs[2])
	eigvals = vcat(oeigvals,eeigvals)
	S::SparseMatrixCSC{Float64, Int64} = cat(sparse(oeigvecs),sparse(eeigvecs); dims=[1,2])
    D::Diagonal{Float64,Vector{Float64}} = Diagonal(eigvals)
	
    left_jumps_in = leftleadops[1:2]
    left_jumps_out = leftleadops[3:4]
    right_jumps_in = rightleadops[1:2]
    right_jumps_out = rightleadops[3:4]

    left_jumps_in2 = map(L->transform_jump_op(D,S'*L*S,TL,μL), left_jumps_in)
    left_jumps_out2 = map(L->transform_jump_op(D,S'*L*S,TL,-μL), left_jumps_out)
    right_jumps_in2 = map(L->transform_jump_op(D,S'*L*S,TR,μR), right_jumps_in)
    right_jumps_out2 = map(L->transform_jump_op(D,S'*L*S,TR,-μR), right_jumps_out)

    superlind = (lindbladian(D, vcat(left_jumps_in2,left_jumps_out2,right_jumps_in2,right_jumps_out2)))
    lindvals,lindvecs = eigsolve(superlind,rand(size(superlind,1)),2,EigSorter(abs;rev=false); tol = 1e-9)
    particle_number = S'*(measureop)*S
    curr1 = real.(map(L->current(first(lindvecs)/(tr(reshape(first(lindvecs),size(S)))), particle_number, superjump(L)), hcat(left_jumps_in2,left_jumps_out2,right_jumps_in2,right_jumps_out2)))
    curr1
end