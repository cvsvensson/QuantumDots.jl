using QuantumDots
using Test

@testset "QuantumDots.jl" begin

end

@testset "Fock" begin
    N = 6
    B = FermionBasis(:🦄)
    focknumber = 20
    fbits = BitVector(bits(focknumber,N))
    ψ = FermionBasisState(focknumber,N,B)
    ψ == FermionBasisState(focknumber,N,B)
    @test focknbr(ψ) == focknumber
    @test chainlength(ψ) == N
    @test bits(ψ) == fbits

    Bspin = FermionBasis{(:↑,:↓)}()
    ψspin = FermionBasisState((:↑=>[1,3],:↓=>[2]),N,Bspin)
    @test [jwstring(Fermion{:↑}(i), ψspin) for i in 1:N] == (-1) .^ [2,2,0,0,0,0]
    @test [jwstring(Fermion{:↓}(i), ψspin) for i in 1:N] == (-1) .^ [2,1,0,0,0,0]

end

@testset "Operators" begin
    N = 2
    B = FermionBasis(:a)
    ψ0 = FermionBasisState(0,N,B)
    CreationOperator(:a,1)
    Cdag1 = CreationOperator(:a,1)
    Cdag2 = CreationOperator{:a}(2)
    newfocknbr, scaling = Cdag1*ψ0
    @test (newfocknbr, scaling) == (1, 1)
    @test bits((Cdag1*ψ0)[1],N) == [1,0]
    newfocknbr, scaling = Cdag2*ψ0
    @test (newfocknbr, scaling) == (2, 1)
    @test bits(newfocknbr,N) == [0,1]

    ψ1 = FermionBasisState(newfocknbr,N,B)
    @test Cdag2*ψ1 == (2,0)
    @test Cdag1*ψ1 == (3,-1)
end

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
    basis = FermionFockBasis(N, species)
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