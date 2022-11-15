module QuantumDots
using LinearAlgebra

export bits,FermionBasis,State,focknbr,chainlength, Fermion, CreationOperator,
    FermionCreationOperator, amplitude, species,states, jwstring,focknbrs,State 

include("fock.jl")
include("operators.jl")
include("state.jl")
# include("symmetry.jl")

end
