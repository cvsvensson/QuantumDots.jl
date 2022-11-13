abstract type AbstractOperator{B<:AbstractBasis} end

struct CreationOperator{B} <: AbstractOperator{B} 
    site::Int
end
focknbr(cdag::CreationOperator{FermionFockBasis})= 2^(cdag.site-1)
# focknbr(cdag::CreationOperator{SpinHalfFockBasis}) where N = 2^(cdag.site-1) * 2^N

#TODO: Implement correct fermion statistics
function Base.:*(cdag::CreationOperator{FermionFockBasis},state::FermionFockBasisState)
    newfocknbr = focknbr(cdag) | focknbr(state)
    amplitude *= count_ones(newfocknbr) == 1 + count_ones(focknbr(state))
    return FermionFockBasisState(newfocknbr,amplitude,chainlength(state))
end