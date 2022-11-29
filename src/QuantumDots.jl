module QuantumDots
using LinearAlgebra
using LinearMaps, SplitApplyCombine

export bits,FermionBasis,State,focknbr,chainlength, Fermion, CreationOperator, particles,
    FermionCreationOperator, amplitude, species,states, jwstring,focknbrs,State 

include("structs.jl")
include("fock.jl")
include("state.jl")
include("operators.jl")
# include("symmetry.jl")

end
