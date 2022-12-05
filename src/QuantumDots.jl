module QuantumDots
using LinearAlgebra, SparseArrays
using LinearMaps, SplitApplyCombine
using BlockDiagonals, Dictionaries

export bits,FermionBasis,State, Fermion, CreationOperator, particles, measure,
    amplitude, jwstring

include("structs.jl")
include("fock.jl")
include("state.jl")
include("operators.jl")
include("symmetry.jl")

end
