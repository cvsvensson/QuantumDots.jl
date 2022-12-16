module QuantumDots
using LinearAlgebra, SparseArrays,LinearMaps, BlockDiagonals, Krylov
using SplitApplyCombine: group, groupreduce
using Dictionaries: sortkeys!
using Symbolics: build_function, @variables

export bits,FermionBasis,FermionParityBasis,State, Fermion, CreationOperator, particles, ParityOperator

include("structs.jl")
include("fock.jl")
include("state.jl")
include("operators.jl")
include("symmetry.jl")
include("symbolic.jl")
include("lindblad.jl")

end
