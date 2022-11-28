module QuantumDots
using LinearAlgebra
using LinearMaps, SplitApplyCombine

export bits,FermionBasis,State,focknbr,chainlength, Fermion, AnnihilationOperator, particles,
    FermionCreationOperator, amplitude, species,states, jwstring,focknbrs,State 

include("fock.jl")
include("state.jl")
include("operators.jl")
# include("symmetry.jl")

end
