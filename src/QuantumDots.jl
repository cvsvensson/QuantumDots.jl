module QuantumDots
using LinearAlgebra, SparseArrays
using TupleTools

export bits,FermionBasis,FermionBasisState,State,focknbr,chainlength, Fermion,
    CreationOperator, amplitude, species,states, jwstring,focknbrs,FermionState 

include("fock.jl")
include("operators.jl")
include("state.jl")
include("symmetry.jl")

end
