using QuantumDots
using Test, LinearAlgebra, SparseArrays

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

@testset "ToggleFermions" begin
    focknbr = 177 # = 1000 1101, msb to the right
    digitpositions = Vector([7, 8, 2, 3])
    daggers = BitVector([1,0,1,1])
    newfocknbr, sign = QuantumDots.togglefermions(digitpositions, daggers, focknbr)
    @test newfocknbr == 119 # = 1110 1110
    @test sign == -1
    # swap two operators
    digitpositions = Vector([7, 2, 8, 3])
    daggers = BitVector([1,1,0,1])
    newfocknbr, sign = QuantumDots.togglefermions(digitpositions, daggers, focknbr)
    @test newfocknbr == 119 # = 1110 1110
    @test sign == 1

    # annihilate twice
    digitpositions = Vector([5, 3, 5])
    daggers = BitVector([0, 1, 0])
    _, sign = QuantumDots.togglefermions(digitpositions, daggers, focknbr)
    @test sign == 0
end

@testset "State" begin
    N = 6
    basis = FermionBasis(N,:a)
    v = rand(length(basis))
    ψ = State(v,basis)
    as = particles(basis)
    @test as[1]*ψ isa typeof(ψ)
    @test eltype(ψ) == Float64
    @test eltype(similar(ψ,Int)) == Int
    # ψrand = rand(FermionState,basis,Float64)
    @test norm(ψ)^2 ≈ ψ'*ψ
    ψsparse = State(sparse(v),basis)
    @test norm(ψsparse)^2 ≈ ψsparse'*ψsparse
end

@testset "Operators" begin
    N = 2
    basis = FermionBasis(N,:a)
    fermions = particles(basis)
    Cdag1 =  FermionCreationOperator((:a,1),basis)
    @test Cdag1.op == fermions[1]'
    Cdag2 = fermions[2]'
    ψ = rand(State,basis,Float64)
    @test Cdag1 * ψ isa State
    @test Cdag1 * State(sparse(vec(ψ)),basis) isa State
    @test Cdag2 * ψ isa State
    @test Cdag2 * State(sparse(vec(ψ)),basis) isa State

    opsum = 2.0Cdag1 - 1.2Cdag1
    @test opsum isa QuantumDots.FockOperatorSum
    @test QuantumDots.imagebasis(opsum) == QuantumDots.imagebasis(Cdag1)
    @test QuantumDots.preimagebasis(opsum) == QuantumDots.preimagebasis(Cdag1)
    @test opsum * ψ ≈  2.0Cdag1*ψ - 1.2Cdag1*ψ
    @test opsum*ψ ≈ .8*Cdag1*ψ
    opsum2 = 2.0Cdag1 - 1.2Cdag2
    @test opsum2 * ψ ≈  2.0Cdag1*ψ - 1.2Cdag2*ψ
    opsum2squared = opsum2*opsum2
    @test opsum2squared * ψ ≈  0*ψ
end


@testset "Hamiltonian" begin
    N = 2
    basis = FermionBasis(N,:a)
    a1,a2 = particles(basis)
    ham = a1'*a1 + π*a2'*a2
    hamwithbasis = basis*ham*basis
    ψ = rand(State,basis,Float64)
    mat = Matrix(hamwithbasis)
    vals,vecs = eigen(mat) 
    @test vals ≈ [0,1,π,π+1]
    parityop = QuantumDots.ParityOperator()
    @test all([State(v,basis)'* parityop * State(v,basis) for v in eachcol(vecs)] .≈ [1,-1,-1,1])
end

@testset "OperatorProduct" begin
    N = 2
    basis = FermionBasis(N,:a)
    a1,a2 = particles(basis)
    parityop = QuantumDots.ParityOperator()
    @test a1*a1 isa CreationOperator
    @test a1*a1*a2' isa CreationOperator
    Fa1 = QuantumDots.FockOperator(a1,basis,basis)
    @test (Fa1*Fa1).op isa CreationOperator
    @test (1*a1).operators[1] isa CreationOperator
    
    @test a1*parityop isa QuantumDots.FockOperatorProduct 
    @test Fa1*parityop isa QuantumDots.FockOperatorProduct 
    @test 1*a1 isa QuantumDots.FockOperatorSum
    @test 1*(Fa1) isa QuantumDots.FockOperatorSum 
    @test 1*parityop isa QuantumDots.FockOperatorSum

    v = State(rand(length(basis)),basis)
    @test (parityop*a1)*v == parityop*(a1*v)
    @test (1*parityop*a1)*v == parityop*(1*a1*v)


    @test eltype(a1) == Int
    @test eltype(parityop) == Int
    @test eltype(Fa1) == Int
    @test eltype(a1*a1) == Int
    @test eltype(1*a1) == Int
    @test eltype(1*Fa1) == Int
    @test eltype(a1+a1) == Int
    @test eltype(1.0*a1) == Float64
    @test eltype(1.0*Fa1) == Float64
    @test eltype(1.0a1+a1) == Float64
    @test eltype(a1+1.0a1) == Float64
    @test eltype(1.0Fa1+a1) == Float64
    @test eltype(a1+1.0Fa1) == Float64
    @test eltype(parityop*(a1+1.0Fa1)) == Float64
end

@testset "Paritybasis and conversions" begin
    N = 2
    basis = FermionBasis(N,:a)
    pbasis = QuantumDots.FermionParityBasis(basis)
    a1,a2 = particles(basis)
    ham = a1'*a1 + π*a2'*a2 + a1'a2
    hamwithbasis = pbasis*ham*pbasis
    lm = QuantumDots.LinearMap(hamwithbasis)
    mat = Matrix(hamwithbasis)
    matlm = Matrix(lm)
    @test mat ≈ matlm
    sp = sparse(hamwithbasis)
    splm = sparse(lm)
    @test sp ≈ splm
    bd = QuantumDots.BlockDiagonal(hamwithbasis)
    bdvals,_ = eigen(bd) 
    spvals,_ = eigen(Matrix(sp)) 
    matvals,_ = eigen(mat) 
    @test bdvals ≈ spvals ≈ matvals
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
        hamiltonian += Δ*Pairing((i,:↑),(i,:↓)) + hc() #Superconductive pairing
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
