module QuantumDots
using LinearAlgebra

export bits,FermionFockBasis,ManyFermionsFockBasis,ManyFermionsFockBasisState,FermionFockBasisState,State,focknbr,chainlength,
    CreationOperator, amplitude, species,states

include("fock.jl")
include("operators.jl")

end
