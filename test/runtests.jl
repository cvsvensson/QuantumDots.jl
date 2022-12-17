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
    function get_ops(qn)
        N = 2
        a = FermionBasis(1:N; qn)
        ham = a[1]'*a[1] + π*a[2]'*a[2]
        vals,vecs = eigen(Matrix(ham))
        @test vals ≈ [0,1,π,π+1]
        parityop = parityoperator(a)
        numberop = numberoperator(a)
        @test all([v'* parityop * v for v in eachcol(vecs)] .≈ [1,-1,-1,1])
        @test Diagonal(diag(parityop)) == parityop
        @test Diagonal(diag(numberop)) == numberop
        @test sum(a[i]'a[i] for i in 1:N) == numberop
        @test prod(2(a[i]'a[i] - 1/2*sparse(I,2^N,2^N)) for i in 1:N) == parityop
        return parityop,numberop
    end
    parityop, numberop = get_ops(QuantumDots.NoSymmetry())
    @test diag(parityop) == [1,-1,-1,1]
    @test diag(numberop) == [0,1,1,2]
    
    parityop, numberop = get_ops(QuantumDots.parity)
    @test diag(parityop) == [-1,-1,1,1]
    @test diag(numberop) == [1,1,0,2]
    
    parityop, numberop = get_ops(QuantumDots.fermionnumber)
    @test diag(parityop) == [1,-1,-1,1]
    @test diag(numberop) == [0,1,1,2]

end


@testset "Fast generated hamiltonians" begin
    N = 5
    params = rand(3)
    hamiltonian(μ,t,Δ; qn) = (a = FermionBasis(1:N; qn); μ*sum(a[i]'a[i] for i in 1:N) + t*(a[1]'a[2] + a[2]'a[1]) + Δ*(a[1]'a[2]' + a[2]a[1]))
    normalham(ps...) = hamiltonian(ps...; qn = QuantumDots.NoSymmetry())
    _, fastham! = QuantumDots.generate_fastham(normalham,:μ,:t,:Δ);
    # @test fastham([1.0,1.0,1.0]) ≈ vec(generator(1.0,1.0,1.0))
    mat = normalham((2 .* params)...)
    fastham!(mat,params)
    @test mat ≈ normalham(params...)
    
    #parity conservation
    parityham(ps...) = hamiltonian(ps...; qn = QuantumDots.parity)
    parityham! = QuantumDots.generate_fastham(parityham,:μ,:t,:Δ)[2];
    mat = parityham((2 .* params)...)
    parityham!(mat,params)
    @test mat ≈ parityham(params...)
    
    #number conservation
    numberham(ps...) = hamiltonian(ps...; qn = QuantumDots.fermionnumber)
    numberham! = QuantumDots.generate_fastham(numberham,:μ,:t,:Δ)[2];
    mat = numberham((2 .* params)...)
    numberham!(mat,params)
    @test mat ≈ numberham(params...)
end

@testset "transport" begin
    N = 1
    a = FermionBasis(N)
    hamiltonian(μ) = μ*sum(a[i]'a[i] for i in 1:N)
    T = rand()
    μL,μR,μH = rand(3)
    jumpinL = a[1]'
    jumpoutL = a[1]
    jumpinR = a[N]'
    jumpoutR = a[N]
    leftlead = QuantumDots.NormalLead(T,μL,jumpinL,jumpoutL)
    rightlead = QuantumDots.NormalLead(T,μR,jumpinR,jumpoutR)
    particle_number = sum(a[i]'a[i] for i in 1:N)
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
