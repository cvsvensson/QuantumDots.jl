module QuantumDots
using LinearAlgebra, SparseArrays, LinearMaps, BlockDiagonals
using SplitApplyCombine: group, sortkeys!
using Dictionaries: dictionary, Dictionary
import FillArrays: Eye, Zeros
using StaticArrays
using SkewLinearAlgebra
import OrderedCollections: OrderedDict
import DifferentiationInterface
using TestItems
using TermInterface

import AxisKeys
import Crayons

using SciMLBase
import SciMLBase: LinearSolution, ODEProblem, ODESolution, solve, solve!, init, LinearProblem, MatrixOperator

export FockNumber, JordanWignerOrdering, bits, FermionBasis,  blockdiagonal, parameter, hc, diagonalize, majorana_coefficients, majorana_polarization, focknumbers
export FockHilbertSpace, SymmetricFockHilbertSpace
export parityoperator, numberoperator, fermions

export qns, pretty_print
export FermionBdGBasis, one_particle_density_matrix, BdGMatrix
export tomatrix, StationaryStateProblem, LindbladSystem, conductance_matrix, PauliSystem, LazyLindbladSystem, NormalLead, CombinedLead
export partial_trace, fermionic_kron, wedge, fermionic_embedding, many_body_density_matrix
export QubitBasis
export @fermions, @majoranas
export FermionConservation, NoSymmetry, ParityConservation, IndexConservation
export project_on_parity, project_on_parities

function fastgenerator end
function fastblockdiagonal end
function TSL_generator end
function fermion_to_majorana end
function majorana_to_fermion end

include("structs.jl")
include("Fock/fock.jl")
include("Fock/phase_factors.jl")
include("Fock/tensor_product.jl")
include("Fock/hilbert_space.jl")
include("Fock/symmetry.jl")
include("Fock/operators.jl")

include("hamiltonians.jl")
include("bdg.jl")
include("diagonalization.jl")
include("qubit.jl")
include("pretty_print.jl")
include("fermionic_tensor_product.jl")
include("reshape.jl")

include("Open system/opensystems.jl")
include("Open system/vectorizers.jl")
include("Open system/lindblad.jl")
include("Open system/lazy_lindblad.jl")
include("Open system/rate_equation.jl")
include("Open system/khatri_rao.jl")
include("Open system/ad.jl")

include("symbolics/muladd.jl")
include("symbolics/symbolic_fermions.jl")
include("symbolics/symbolic_majoranas.jl")


import PrecompileTools

PrecompileTools.@compile_workload begin
    # c1 = FermionBasis(1:2)
    # c2 = FermionBasis(1:1, (:s,); qn=ParityConservation())
    # blockdiagonal(c2[1, :s], c2)
    # c3 = FermionBasis(3:3; qn=FermionConservation())
    # partial_trace(rand(2^2), c1, FermionBasis(1:1))
    # c = wedge(c1, c3)
    # cs = (c1, c3)
    # reshape(wedge((first(c1), first(c3)), cs, c), c, cs)
    # cbdg = FermionBdGBasis(1:1, (:s,))
    # BdGMatrix(cbdg[1, :s]' * cbdg[1, :s])
    # @fermions f
    # QuantumDots.eval_in_basis((f[1] * f[2]' + 1 + f[1])^2, c1)
    # @majoranas γ
    # (γ[1] * γ[2] + 1 + γ[1])^2
end

end
