using QuantumDots
using Test

@testset "QuantumDots.jl" begin

end

@testset "Fock" begin
    N = 6
    B = FermionFockBasis{:🦄}()
    focknumber = 20
    fbits = BitVector(bits(focknumber,N))
    ψ = FermionBasisState(focknumber,N,B)
    ψ == FermionBasisState{:🦄}(focknumber,N)
    @test focknbr(ψ) == focknumber
    @test chainlength(ψ) == N
    @test bits(ψ)[:🦄] == fbits

    Bspin = FermionFockBasis{(:↑,:↓)}()
    ψspin = FermionBasisState((focknumber,focknumber),N,Bspin)
    Bspin2 = FermionFockBasis{(:↑,)}()
    ψspin2 = FermionBasisState((focknumber,),2*N,Bspin2)
    #ψspin takes up more memory than ψspin2. Should switch to using a single focknumber. 

    # @test focknbr(ψspin) == focknbr(ψ) + focknbr(ψ)*2^N 
    # @test chainlength(ψspin) == N
end

@testset "Operators" begin
    N = 2
    B = FermionFockBasis(:a)
    ψ0 = FermionFockBasisState(0,N,B)
    Cdag1 = CreationOperator{B}(1)
    Cdag2 = CreationOperator{B}(2)
    @test focknbr(Cdag1*ψ0) == 1
    @test bits(Cdag1*ψ0) == [1,0]
    @test focknbr(Cdag2*ψ0) == 2
    @test bits(Cdag2*ψ0) == [0,1]
    @test focknbr(Cdag2*(Cdag2*ψ0)) isa Missing
end

@testset "interface" begin
    # We want an interface something like this
    species = :↑,:↓
    N = 4
    basis = FermionFockBasis(N, species; conserve_parity=false)

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