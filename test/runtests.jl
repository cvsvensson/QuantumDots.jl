using QuantumDots
using Test, LinearAlgebra

@testset "QuantumDots.jl" begin

end

@testset "Fock" begin
    N = 6
    B = FermionBasis(N,:🦄)
    focknumber = 20 # = 16+4 = 00101
    fbits = bits(focknumber,N)
    @test fbits == [0,0,1,0,1,0]
    Bspin = FermionBasis(N,(:↑,:↓))
    @test length(particles(Bspin)) == 2*N
    @test length(Bspin) == 2^(2N)
end

@testset "State" begin
    N = 6
    basis = FermionBasis(N,:a)
    v = rand(length(basis))
    ψ = State(v,basis)
    using SparseArrays
    ψsparse = State(sparse(v),basis)

    # ψrand = rand(FermionState,basis,Float64)
    @test norm(ψ)^2 ≈ ψ'*ψ
end

@testset "Operators" begin
    N = 2
    basis = FermionBasis(N,:a)
    Cdag1 = FermionCreationOperator((:a,1),basis)
    ψ = rand(State,basis,Float64)
    @test Cdag1 * ψ isa State
    @test Cdag1 * State(sparse(vec(ψ)),basis) isa State
end

wish = false
if wish == true 
    @testset "interface" begin
    # We want an interface something like this
    species = :↑,:↓
    N = 4
    basis = FermionBasis(N, species; conserve_parity=false)

    ψ = randomstate(basis) #Dense or sparse?
    @test chainlength(ψ) == N
    @test eltype(ψ) == Float64
    @test inner(ψ,ψ) ≈ 1 #Norm of state
    ψ' # Lazy adjoint. Need to implement support for Adjoint{T,State}
    
    @test ψ' * ψ == inner(ψ,ψ) == inner(ψ',ψ)
    outer(ψ,ψ) == ψ*ψ'
    ψf = randomfockstate(basis) #A particular fock state. Should be sparse
    ψfdense = vec(ψf) #Dense version


    hamiltonian = emptyoperator(basis)
    #Intra site
    for i in 1:N
        hamiltonian += Δ*Pairing(i,i,:↑,:↓) + hc() #Superconductive pairing
        hamiltonian += U*NumberOperator(i,:↑)*NumberOperator(i,:↓) #Coulomb interaction
    end
    #Inter site
    for i in 1:N-1
        hamiltonian += t*Hopping(i,i+1,:↑) + hc() #Standard hopping. Maybe Hopping((i,:↑),(i+1,:↑)) also or instead
        hamiltonian += t*Hopping(i,i+1,:↓) + hc()
        hamiltonian += α*Hopping(i,i+1,:↑,:↓) + hc() #Spin orbit
        hamiltonian += α*Hopping(i,i+1,:↓,:↑) + hc() 
        hamiltonian += Δ1*Pairing(i,i+1,:↑,:↓) + hc() #Superconductive pairing
        hamiltonian += Δ1*Pairing(i,i+1,:↓,:↑) + hc()
        hamiltonian += V*NumberOperator(i)*NumberOperator(i+1) #Coulomb
    end

    eigsolve(hamiltonian) #Iterative eigensolver, which does not rely on the hamiltonian being dense. Can give a few of the lowest energy states
    Hmat = Matrix(hamiltonian,basis) #Convert to standard dense matrix.
    eigen(Hmat) #Exact diagonalization

    ##Any symbol can be used for species
    hamiltonian = emptyoperator()
    species = :🦄,:👹 #(\unicorn_face, \japanese_ogre) 
    N = 2
    basis = FermionBasis(N, species)
    ## Can write hamiltonian in terms of elementary fermionic operators as well
    #Intra site
    for i in 1:N
        hamiltonian += Δ*Cdag(i,:👹)*Cdag(i,:🦄) + hc() #Superconductive pairing
        hamiltonian += U*Cdag(i,:👹)*C(i,:👹)*Cdag(i,:🦄)*C(i,:🦄) #Coulomb interaction
    end
    #Inter site
    for i in 1:N-1
        hamiltonian += t*Cdag(i+1,:🦄)*C(i,:🦄) + hc() #Standard hopping. Maybe Hopping((i,:↑),(i+1,:↑)) also or instead
        hamiltonian += t*Cdag(i+1,:👹)*C(i,:👹) + hc()
        hamiltonian += α*Cdag(i,:🦄)*C(i+1,:) + hc() #Spin orbit
        hamiltonian += α*Cdag(i,:👹)*C(i+1,:🦄) + hc() 
        hamiltonian += Δ1*Cdag(i,:👹)*Cdag(i+1,:🦄) + hc() #Superconductive pairing
        hamiltonian += Δ1*Cdag(i,:🦄)*Cdag(i+1,:👹) + hc() #Superconductive pairing
        hamiltonian += V*(Cdag(i,:👹)*C(i,:👹) + Cdag(i,:🦄)*C(i,:🦄)) * (Cdag(i+1,:👹)*C(i+1,:👹) + Cdag(i+1,:🦄)*C(i+1,:🦄)) #Coulomb
    end
end
end