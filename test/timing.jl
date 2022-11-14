#This function compares standard matrix mult vs the bitwise strategy. Similar memory alloc. Bitwise faster at N>6
function timetest(N)
    B = FermionBasis(N,:a)
    ψrand = rand(FermionState,B,Float64)
    println("Fock")
    focktime(N,ψrand)
    println("Dense")
    densetime(N,ψrand)
end

function densetime(N,ψrand)
    M = rand(2^N,2^N)
    @time out1 = M*vec(ψrand)
    return
end
function focktime(N,ψrand)
    op = FermionCreationOperator(:a1)
    op*ψrand
    @time out2 = op*ψrand
    return
end