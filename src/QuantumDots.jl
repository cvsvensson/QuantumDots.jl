module QuantumDots
using LinearAlgebra

export bits,SpinHalfFockBasis,SpinlessFockBasis,SpinHalfFockBasisState,SpinlessFockBasisState,State,focknbr,chainlength,
    CreationOperator

include("fock.jl")
include("operators.jl")

end
