abstract type AbstractOperator{B<:AbstractBasis} end

struct CreationOperator{B} <: AbstractOperator{B} 
    site::Int
end
focknbr(cdag::CreationOperator{SpinlessFockBasis{N}}) where N = 2^(cdag.site-1)
focknbr(cdag::CreationOperator{SpinHalfFockBasis{N}}) where N = 2^(cdag.site-1) * 2^N

#TODO: Implement correct fermion statistics
function Base.:*(cdag::CreationOperator{SpinlessFockBasis{N}},state::SpinlessFockBasisState{N}) where N
    newfocknbr = focknbr(cdag) | focknbr(state)
    if count_ones(newfocknbr) == 1 + count_ones(focknbr(state))
        return SpinlessFockBasisState{N}(newfocknbr)
    else 
        return SpinlessFockBasisState{N}(missing)
    end
end