#This function compares standard matrix mult vs the bitwise strategy. Similar memory alloc. Bitwise faster at N>6
using BenchmarkTools
function timetest(N)
    basis = FermionBasis(N,:a)
    ψ = rand(State,basis,Float64)
    println("Fock")
    op = FermionCreationOperator((:a,1),basis)
    focktime(op,ψ)
    println("Dense")
    M = rand(2^N,2^N)
    v = vec(ψ)
    densetime(M,v)
end

function densetime(M,v)
    @time out1 = M*v
    return
end
function focktime(op,ψ)
    @time out2 = op*ψ
    return
end