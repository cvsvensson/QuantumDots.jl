module QuantumDots
using LinearAlgebra, SparseArrays
using LinearMaps, SplitApplyCombine
using BlockDiagonals, Dictionaries
using Symbolics
export bits,FermionBasis,FermionParityBasis,State, Fermion, CreationOperator, particles, measure,
    amplitude, jwstring, ParityOperator

include("structs.jl")
include("fock.jl")
include("state.jl")
include("operators.jl")
include("symmetry.jl")
include("symbolic.jl")

end
