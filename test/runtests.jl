using QuantumDots
using Test

@testset "QuantumDots.jl" begin

end

@testset "Fock" begin
    N = 6
    B = FermionFockBasis(:a)
    focknumber = 20
    fbits = BitVector(digits(focknumber, base=2, pad=N))
    ψ = FermionFockBasisState(focknumber,N,B)
    @test focknbr(ψ) == focknumber
    @test chainlength(ψ) == N
    @test bits(ψ) == fbits

    Bspin = ManyFermionsFockBasis(:↑,:↓)
    ψspin = ManyFermionsFockBasisState((fbits,fbits),N,Bspin)
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
