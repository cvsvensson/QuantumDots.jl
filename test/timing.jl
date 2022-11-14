#This function compares standard matrix mult vs the bitwise strategy. Similar memory alloc. Bitwise faster at N>6
function timetest(N)
    M = rand(2^N,2^N)
    ψrand = rand(FermionState{(:a,),Float64},N)
    op = CreationOperator(:a,1)
    M*vec(ψrand)
    op*ψrand
    @time out1 = M*vec(ψrand)
    @time out2 = op*ψrand
    return (out1,out2)
end