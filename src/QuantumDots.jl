module QuantumDots
using LinearAlgebra, SparseArrays, LinearMaps, BlockDiagonals
using Reexport
using SplitApplyCombine: group
using Dictionaries#: sortkeys!
using Symbolics
using StaticArrays
using UnicodePlots
using OrdinaryDiffEq
@reexport import OrdinaryDiffEq: ODEProblem, solve, solve!, init
using LinearSolve
@reexport import LinearSolve: LinearProblem
import SciMLBase: LinearSolution
# import AbstractDifferentiation as AD

export bits, FermionBasis, parityoperator, numberoperator, blockdiagonal, parameter
export qns, Z2, QArray, Z2Symmetry, QNIndex
export one_particle_density_matrix

include("structs.jl")
include("fock.jl")
include("operators.jl")
include("symmetry.jl")
include("lattice.jl")
include("symbolic.jl")
include("opensystems.jl")
include("vectorizers.jl")
include("lindblad.jl")
include("hamiltonians.jl")
include("QArray.jl")
include("bdg.jl")
include("pretty_print.jl")
include("rate_equation.jl")
include("khatri_rao.jl")
end
