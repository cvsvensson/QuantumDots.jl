using QuantumDots
using Test

@testset "QuantumDots.jl" begin

end

@testset "Fock" begin
    N = 6
    B = SpinlessFockBasis{N}
    focknumber = 20
    fbits = BitVector(digits(focknumber, base=2, pad=N))
    ψ = SpinlessFockBasisState{N}(focknumber)
    @test focknbr(ψ) == focknumber
    @test chainlength(ψ) == N
    @test bits(ψ) == fbits
    ψspin = SpinHalfFockBasisState(ψ,ψ)
    @test focknbr(ψspin) == focknbr(ψ) + focknbr(ψ)*2^N 
    @test chainlength(ψspin) == N
end

@testset "Operators" begin
    N = 2
    B = SpinlessFockBasis{N}
    ψ0 = SpinlessFockBasisState{N}(0)
    Cdag1 = CreationOperator{B}(1)
    Cdag2 = CreationOperator{B}(2)
    @test focknbr(Cdag1*ψ0) == 1
    @test bits(Cdag1*ψ0) == [1,0]
    @test focknbr(Cdag2*ψ0) == 2
    @test bits(Cdag2*ψ0) == [0,1]
    @test focknbr(Cdag2*(Cdag2*ψ0)) isa Missing
end
