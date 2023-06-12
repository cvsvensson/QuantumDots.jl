module QuantumDots
using LinearAlgebra, SparseArrays,LinearMaps, BlockDiagonals, Krylov
using SplitApplyCombine: group
using Dictionaries#: sortkeys!
using Symbolics
using StaticArrays
using UnicodePlots
using LinearSolve

export bits,FermionBasis, parityoperator, numberoperator, blockdiagonal, parameter
export qns,Z2,QArray, Z2Symmetry, QNIndex
export one_particle_density_matrix

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
include("rate_equation.jl")
end
