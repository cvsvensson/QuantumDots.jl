module QuantumDots
using LinearAlgebra, SparseArrays, LinearMaps, BlockDiagonals
using SplitApplyCombine: group, sortkeys!
using Dictionaries: dictionary, Dictionary
import FillArrays: Eye
using StaticArrays
using SkewLinearAlgebra
import OrderedCollections: OrderedDict
import AbstractDifferentiation as AD
using TestItems
using TermInterface

import AxisKeys
import Crayons

using SciMLBase
import SciMLBase: LinearSolution, ODEProblem, ODESolution, solve, solve!, init, LinearProblem, MatrixOperator

export FockNumber, JordanWignerOrdering, bits, FermionBasis, parityoperator, numberoperator, blockdiagonal, parameter, hc, diagonalize, majorana_coefficients, majorana_polarization
export qns, pretty_print
export FermionBdGBasis, one_particle_density_matrix, BdGMatrix
export tomatrix, StationaryStateProblem, LindbladSystem, conductance_matrix, PauliSystem, LazyLindbladSystem, NormalLead, CombinedLead
export partial_trace, wedge, many_body_density_matrix
export QubitBasis
export @fermions, @majoranas
export FermionConservation, NoSymmetry, ParityConservation, IndexConservation

function fastgenerator end
function fastblockdiagonal end
function TSL_generator end
function fermion_to_majorana end
function majorana_to_fermion end

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
include("symbolics/muladd.jl")
include("symbolics/symbolic_fermions.jl")
include("symbolics/symbolic_majoranas.jl")


import PrecompileTools

PrecompileTools.@compile_workload begin
    c1 = FermionBasis(1:2)
    c2 = FermionBasis(1:1, (:s,); qn=ParityConservation())
    blockdiagonal(c2[1, :s], c2)
    c3 = FermionBasis(2:2; qn=FermionConservation())
    vals, vecs = eigen(Matrix(c1[1]' * c1[1]))
    partial_trace(vecs[1, :], (1,), c1)
    wedge(c1, c2)
    cbdg = FermionBdGBasis(1:1, (:s,))
    diagonalize(BdGMatrix(cbdg[1, :s]' * cbdg[1, :s]))
    @fermions f
    QuantumDots.eval_in_basis((f[1] * f[2]' + 1 + f[1])^2, c1)
    @majoranas γ
    (γ[1] * γ[2] + 1 + γ[1])^2
end

end
