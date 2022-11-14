abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end
symmetry(::FermionBasis) = NoSymmetry()
symmetry(::Type{<:FermionBasis}) = NoSymmetry()
focknbr(ind::Integer,::Val{S},::NoSymmetry) where S = ind-1
