module QuantumDots
using LinearAlgebra, SparseArrays

export bits,FermionBasis,State,focknbr,chainlength, Fermion,
    FermionCreationOperator, amplitude, species,states, jwstring,focknbrs,FermionState 

include("fock.jl")
include("operators.jl")
include("state.jl")
include("symmetry.jl")

end
