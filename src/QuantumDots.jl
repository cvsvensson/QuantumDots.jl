module QuantumDots
using LinearAlgebra, SparseArrays, LinearMaps, BlockDiagonals
using Reexport
using SplitApplyCombine: group
using Dictionaries
using StaticArrays
using UnicodePlots

using SciMLBase
import SciMLBase: LinearSolution, ODEProblem, ODESolution, solve, solve!, init, LinearProblem, MatrixOperator

export bits, FermionBasis, parityoperator, numberoperator, blockdiagonal, parameter
export qns, Z2, QArray, Z2Symmetry, QNIndex
export one_particle_density_matrix
export tomatrix, StationaryStateProblem, Lindbladsystem, conductance_matrix

function fastgenerator end
function fastblockdiagonal end
function TSL_generator end
function chem_derivative end

include("structs.jl")
include("fock.jl")
include("operators.jl")
include("symmetry.jl")
include("lattice.jl")
include("opensystems.jl")
include("vectorizers.jl")
include("lindblad.jl")
include("lazy_lindblad.jl")
include("hamiltonians.jl")
include("QArray.jl")
include("bdg.jl")
include("pretty_print.jl")
include("rate_equation.jl")
include("khatri_rao.jl")
end
