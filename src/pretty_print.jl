function Base.print(v::AbstractVector, b::FermionBasis{N}; digits = 3) where N
    println("labels = |", Tuple(keys(b))...,">")
    for qn in qns(b)
        println("QN = ",qn)
        for ind in qntoinds(qn,b)
            fs = indtofock(ind,b)
            print(" |",Int.(bits(fs,N))...,">")
            println(" : ", round(v[ind];digits))
        end
    end
end

qns(b::FermionBasis) = qns(b.symmetry)
qns(::NoSymmetry) = (nothing,)
qns(sym::AbelianFockSymmetry) = Tuple(keys(sym.qntoinds))

qntoinds(::Nothing,::FermionBasis{M,<:Any,<:Any,NoSymmetry}) where M = 1:2^M
qntoinds(qn,b::FermionBasis) = b.symmetry.qntoinds[qn]