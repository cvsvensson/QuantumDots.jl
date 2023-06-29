module QuantumDots
using LinearAlgebra, SparseArrays, LinearMaps, BlockDiagonals
using Reexport
using SplitApplyCombine: group
using Dictionaries
using Symbolics
using StaticArrays
using UnicodePlots

using SciMLBase
import SciMLBase: LinearSolution, ODEProblem, ODESolution, solve, solve!, init, LinearProblem

export bits, FermionBasis, parityoperator, numberoperator, blockdiagonal, parameter
export qns, Z2, QArray, Z2Symmetry, QNIndex
export one_particle_density_matrix
export tomatrix

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
