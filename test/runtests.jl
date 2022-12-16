using QuantumDots
using Test, LinearAlgebra, SparseArrays, Random, Krylov
Random.seed!(1234)

@testset "QuantumDots.jl" begin

end

@testset "Fock" begin
    N = 6
    focknumber = 20 # = 16+4 = 00101
    fbits = bits(focknumber,N)
    @test fbits == [0,0,1,0,1,0]

    @testset "removefermion" begin
        focknbr = rand(1:2^N) - 1
        fockbits = bits(focknbr,N)
        function test_remove(n)
            QuantumDots.removefermion(n,focknbr) == (fockbits[n] ? (focknbr - 2^(n-1), (-1)^sum(fockbits[1:n-1])) : (0,0))
        end
        all([test_remove(n) for n in 1:N])
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
end

@testset "Basis" begin
    N = 6
    B = FermionBasis(1:N)
    @test QuantumDots.nbr_of_fermions(B) == N
    Bspin = FermionBasis(1:N,(:↑,:↓))
    @test QuantumDots.nbr_of_fermions(Bspin) == 2N
    @test B[1] isa SparseMatrixCSC
    @test Bspin[1,:↑] isa SparseMatrixCSC
    @test parityoperator(B) isa SparseMatrixCSC
    @test parityoperator(Bspin) isa SparseMatrixCSC
end

@testset "Hamiltonian" begin
    N = 2
    a = FermionBasis(1:N)
    ham = a[1]'*a[1] + π*a[2]'*a[2]
    vals,vecs = eigen(Matrix(ham))
    @test vals ≈ [0,1,π,π+1]
    parityop = parityoperator(a)
    @test all([v'* parityop * v for v in eachcol(vecs)] .≈ [1,-1,-1,1])
end

@testset "Paritybasis and conversions" begin
    N = 2
    🦄 = FermionBasis(1:N)
    pbasis = FermionBasis(1:N; qn = QuantumDots.parity)
    pbasis = FermionParityBasis(basis)
    ham = 🦄[1]'*🦄[1] + π*🦄[2]'*🦄[2] + 🦄[1]'🦄[2]
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

@testset "Fast generated hamiltonians" begin
    N = 5
    basis = FermionBasis(N,symbol=:a)
    a = particles(basis)
    hamiltonian(μ,t,Δ) = μ*sum(a[i]'a[i] for i in 1:N) + t*(a[1]'a[2] + a[2]'a[1]) + Δ*(a[1]'a[2]' + a[2]a[1])
    matrixgenerator(μ, t,Δ) = Matrix(basis*hamiltonian(μ,t,Δ)*basis)
    _, fastham! = QuantumDots.generate_fastham(matrixgenerator,:μ,:t,:Δ);
    # @test fastham([1.0,1.0,1.0]) ≈ vec(generator(1.0,1.0,1.0))
    mat = zero(matrixgenerator(1.0,1.0,1.0))
    fastham!(mat,[1.0,1.0,1.0])
    @test mat ≈ matrixgenerator(1,1,1)
    
    #parity conservation
    basis = FermionParityBasis(basis)
    generator(μ, t,Δ) = QuantumDots.BlockDiagonal(basis*hamiltonian(μ,t,Δ)*basis).blocks
    oddham! = QuantumDots.generate_fastham(first ∘ generator,:μ,:t,:Δ)[2];
    evenham! = QuantumDots.generate_fastham(last ∘ generator ,:μ,:t,:Δ)[2];
    mateven = zero(first(generator(1.0,2.0,3.0)))
    matodd = zero(last(generator(1.0,2.0,3.0)))
    oddham!(matodd,[1.0,2.0,3.0])
    @test matodd ≈ first(generator(1,2,3))
    evenham!(mateven,[1.0,2.0,3.0])
    @test mateven ≈ last(generator(1,2,3))

    generatorsp(μ, t,Δ) = QuantumDots.spBlockDiagonal(basis*hamiltonian(μ,t,Δ)*basis).blocks
    oddhamsp! = QuantumDots.generate_fastham(first ∘ generatorsp ,:μ,:t,:Δ)[2];
    evenhamsp! = QuantumDots.generate_fastham(last ∘ generatorsp ,:μ,:t,:Δ)[2];
    matoddsp = sparse(first(generatorsp(1.0,2.0,3.0))) #For correct sparsity structure
    matevensp = sparse(last(generatorsp(1.0,2.0,3.0)))
    oddham!(matoddsp,[1.0,2.0,3.0])
    @test matoddsp ≈ first(generatorsp(1,2,3))
    evenham!(matevensp,[1.0,2.0,3.0])
    @test matevensp ≈ last(generatorsp(1,2,3))

    @test matevensp ≈ mateven
    @test matodd ≈ matodd
end

@testset "transport" begin
    N = 1
    basis = FermionBasis(N,symbol=:a)
    a = particles(basis)
    hamiltonian(μ) = sparse((μ*sum(a[i]'a[i] for i in 1:N)),basis,basis)
    T = rand()
    μL,μR,μH = rand(3)
    jumpinL = sparse(a[1]',basis,basis)
    jumpoutL = sparse(a[1],basis,basis)
    jumpinR = sparse(a[N]',basis,basis)
    jumpoutR = sparse(a[N],basis,basis)
    leftlead = QuantumDots.NormalLead(T,μL,jumpinL,jumpoutL)
    rightlead = QuantumDots.NormalLead(T,μR,jumpinR,jumpoutR)
    particle_number = sparse(sum(a[i]'a[i] for i in 1:N),basis,basis)
    system = QuantumDots.OpenSystem(hamiltonian(μH),[leftlead, rightlead])

    diagonalsystem = QuantumDots.diagonalize(system)
    transformedsystem = QuantumDots.ratetransform(diagonalsystem)
    superjumpins = QuantumDots.dissipator.(QuantumDots.jumpins(transformedsystem))
    superjumpouts = QuantumDots.dissipator.(QuantumDots.jumpouts(transformedsystem))
    superlind = QuantumDots.lindbladian(Diagonal(QuantumDots.eigenvalues(transformedsystem)), vcat(superjumpins,superjumpouts))
    solver = LsmrSolver(4^N+1,4^N,Vector{ComplexF64})
    ρ = QuantumDots.stationary_state(superlind; solver)
    rhom = reshape(ρ,2^N,2^N)
    rhod = diag(rhom)
    p2 = (QuantumDots.fermidirac(μH,T,μL) + QuantumDots.fermidirac(μH,T,μR))/2
    p1 = 1 - p2
    analytic_current = -1/2*(QuantumDots.fermidirac(μH,T,μL) - QuantumDots.fermidirac(μH,T,μR))
    @test rhod ≈ [p1, p2]

    numeric_current = real.(QuantumDots.conductance(system,[particle_number])[1])
    @test abs(sum(numeric_current)) < 1e-10
    @test sum(numeric_current; dims = 2) ≈ analytic_current .* [-1, 1] #Why not flip the signs?
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
