module QuantumDots
using LinearAlgebra
using LinearMaps

export bits,FermionBasis,State,focknbr,chainlength, Fermion, CreationOperator,
    FermionCreationOperator, amplitude, species,states, jwstring,focknbrs,State 

include("fock.jl")
include("state.jl")
include("operators.jl")
# include("symmetry.jl")

end
