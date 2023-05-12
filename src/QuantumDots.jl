module QuantumDots
using LinearAlgebra, SparseArrays,LinearMaps, BlockDiagonals, Krylov
using SplitApplyCombine: group
using Dictionaries#: sortkeys!
using Symbolics
using StaticArrays
using UnicodePlots

export bits,FermionBasis, parityoperator, numberoperator, blockdiagonal
export qns,Z2,QArray, Z2Symmetry, QNIndex

include("structs.jl")
include("fock.jl")
include("operators.jl")
include("symmetry.jl")
include("lattice.jl")
include("symbolic.jl")
include("lindblad.jl")
include("hamiltonians.jl")
include("QArray.jl")
include("bdg.jl")
include("pretty_print.jl")

end
