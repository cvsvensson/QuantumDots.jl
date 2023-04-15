module QuantumDots
using LinearAlgebra, SparseArrays,LinearMaps, BlockDiagonals, Krylov
using SplitApplyCombine: group
using Dictionaries#: sortkeys!
using Symbolics
using TruncatedStacktraces
using StaticArrays

export bits,FermionBasis, parityoperator, numberoperator, blockdiagonal

include("structs.jl")
include("fock.jl")
include("operators.jl")
include("symmetry.jl")
include("lattice.jl")
include("symbolic.jl")
include("lindblad.jl")
include("hamiltonians.jl")
include("truncate_stacktraces.jl")

end
