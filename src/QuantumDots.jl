module QuantumDots
using LinearAlgebra, SparseArrays, LinearMaps
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

export FockNumber, JordanWignerOrdering, bits, blockdiagonal, hc, diagonalize, focknumbers
export FockHilbertSpace, SymmetricFockHilbertSpace, SimpleFockHilbertSpace, hilbert_space
export parityoperator, numberoperator, fermions, majoranas, matrix_representation

export pretty_print
export FermionBdGBasis, one_particle_density_matrix, BdGMatrix
export tomatrix, StationaryStateProblem, LindbladSystem, conductance_matrix, PauliSystem, LazyLindbladSystem, NormalLead, CombinedLead
export partial_trace, fermionic_kron, wedge, embedding, many_body_density_matrix
export QubitBasis
export @fermions, @majoranas
export FermionConservation, NoSymmetry, ParityConservation, IndexConservation
export project_on_parity, project_on_parities

## Symbolics extension
function fastgenerator end
function fastblockdiagonal end
function fermion_to_majorana end
function majorana_to_fermion end
## BlockDiagonals extension
function blockdiagonal end
function blocks end

## Files
include("structs.jl")
include("Fock/fock.jl")
include("Fock/phase_factors.jl")
include("Fock/hilbert_space.jl")
include("Fock/symmetry.jl")
include("Fock/operators.jl")
include("Fock/tensor_product.jl")

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
    H1 = hilbert_space(1:2)
    H2 = hilbert_space(Base.product(1:2, (:s,)), ParityConservation())
    H3 = hilbert_space(3:3, FermionConservation())
    c1 = fermions(H1)
    c2 = fermions(H2)
    c3 = fermions(H3)
    partial_trace(rand(4, 4) + hc, H1 => hilbert_space(1:1))
    H = wedge(H1, H3)
    Hs = (H1, H3)
    reshape(H => Hs)(wedge(Hs => H)(c1[1], c3[3]))
    cbdg = FermionBdGBasis(1:1, (:s,))
    BdGMatrix(cbdg[1, :s]' * cbdg[1, :s])
    @fermions f
    QuantumDots.eval_in_basis((f[1] * f[2]' + 1 + f[1])^2 * 2.0 + hc, c1)
    matrix_representation((f[1] * f[2]' + 1 + f[1])^2 * 2.0, H1)
    @majoranas γ
    (γ[1] * γ[2] + 1 + γ[1])^2
end

end
