module QuantumDots
using LinearAlgebra, SparseArrays,LinearMaps, BlockDiagonals, Krylov, BlockArrays
using SplitApplyCombine: group
using Dictionaries#: sortkeys!
using Symbolics: build_function, @variables

export bits,FermionBasis,FermionParityBasis, parityoperator

include("structs.jl")
include("fock.jl")
include("operators.jl")
include("symmetry.jl")
include("symbolic.jl")
include("lindblad.jl")

end
