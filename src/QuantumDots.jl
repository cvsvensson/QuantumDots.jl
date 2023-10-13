module QuantumDots
using LinearAlgebra, SparseArrays, LinearMaps, BlockDiagonals
using Reexport
using SplitApplyCombine: group
using Dictionaries
using StaticArrays

using SciMLBase
import SciMLBase: LinearSolution, ODEProblem, ODESolution, solve, solve!, init, LinearProblem, MatrixOperator

export bits, FermionBasis, parityoperator, numberoperator, blockdiagonal, parameter, hc, diagonalize, majorana_coefficients, majorana_polarization
export qns, Z2, QArray, Z2Symmetry, QNIndex, pretty_print
export one_particle_density_matrix, partial_trace
export tomatrix, StationaryStateProblem, Lindbladsystem, conductance_matrix
export QubitBasis

function fastgenerator end
function fastblockdiagonal end
function TSL_generator end
function chem_derivative end
function visualize end
function majvisualize end

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
include("rate_equation.jl")
include("khatri_rao.jl")
include("diagonalization.jl")
include("majorana.jl")
include("qubit.jl")
include("pretty_print.jl")
include("perturbation.jl")

end
