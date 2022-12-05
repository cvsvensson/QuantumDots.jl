module QuantumDots
using LinearAlgebra, SparseArrays
using LinearMaps, SplitApplyCombine
using BlockDiagonals, Dictionaries

export bits,FermionBasis,State,focknbr,chainlength, Fermion, CreationOperator, particles,
    FermionCreationOperator, amplitude, species,states, jwstring,focknbrs,State 

include("structs.jl")
include("fock.jl")
include("state.jl")
include("operators.jl")
include("symmetry.jl")

end