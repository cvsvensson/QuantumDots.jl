module QuantumDots
using LinearAlgebra, SparseArrays, LinearMaps, BlockDiagonals
using Reexport
using SplitApplyCombine: group
using Dictionaries
using StaticArrays
using SkewLinearAlgebra
import OrderedCollections: OrderedDict
import AbstractDifferentiation as AD
using FillArrays: Eye
using TestItems

using SciMLBase
import SciMLBase: LinearSolution, ODEProblem, ODESolution, solve, solve!, init, LinearProblem, MatrixOperator

export bits, FermionBasis, parityoperator, numberoperator, blockdiagonal, parameter, hc, diagonalize, majorana_coefficients, majorana_polarization
export qns, pretty_print
export FermionBdGBasis, one_particle_density_matrix, BdGMatrix
export tomatrix, StationaryStateProblem, LindbladSystem, conductance_matrix, PauliSystem, LazyLindbladSystem, NormalLead, CombinedLead
export partial_trace, wedge, many_body_density_matrix
export QubitBasis
export @fermion
export FermionConservation, NoSymmetry, ParityConservation, IndexConservation

function fastgenerator end
function fastblockdiagonal end
function TSL_generator end

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
include("bdg.jl")
include("rate_equation.jl")
include("khatri_rao.jl")
include("diagonalization.jl")
include("majorana.jl")
include("qubit.jl")
include("pretty_print.jl")
include("ad.jl")
include("wedge.jl")
include("symbolic_fermions.jl")


import PrecompileTools

PrecompileTools.@compile_workload begin
    c1 = FermionBasis(1:1)
    c2 = FermionBasis(1:1, (:s,); qn=QuantumDots.parity)
    blockdiagonal(c2[1, :s], c2)
    c3 = FermionBasis(2:2; qn=QuantumDots.fermionnumber)
    vals, vecs = eigen(Matrix(c1[1]' * c1[1]))
    partial_trace(vecs[1, :], (1,), c1)
    cbdg = FermionBdGBasis(1:1, (:s,))
    diagonalize(BdGMatrix(cbdg[1, :s]' * cbdg[1, :s]))
end

end
