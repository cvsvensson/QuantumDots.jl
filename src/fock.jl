
abstract type AbstractBasis end
abstract type AbstractBasisState{B<:AbstractBasis} end
abstract type AbstractState{B<:AbstractBasisState} end
const DEFAULT_FERMION_SYMBOL = :f

focknbr(bits::Union{BitVector,Vector{Bool}}) = mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))
focknbr(sites::Vector{<:Integer}) = mapreduce(site -> 2^(site-1),+, sites)
focknbr(sites::Vector{<:Integer},cell_length, species_index=1) = mapreduce(site -> 2^(site + cell_length + species_index - 3),+, sites)
bits(s::Integer,N) = digits(Bool,s, base=2, pad=N)
# bits(s::Integer,N) = BitVector(digits(Bool,s, base=2, pad=N)) #more allocations


struct FermionBasis{S} <: AbstractBasis end
FermionBasis() = FermionBasis{(DEFAULT_FERMION_SYMBOL,)}()
FermionBasis(s::Symbol) = FermionBasis{(s,)}()
#FermionFockBasis(args...) = FermionFockBasis{args}() #Should we allow this type unstable constructor?
species(::FermionBasis{S}) where S = S

struct Fermion{S}
    site::Int
end
species(::Fermion{S}) where S = S

struct FermionBasisState{S,M} 
    focknbr::Int
    chainlength::Int
    FermionBasisState{S}(f::Integer,c::Integer) where S = new{S,length(S)}(f,c)
end
FermionBasisState(focknbr,length,::FermionBasis{S}) where S = FermionBasisState{S}(focknbr,length)
FermionBasisState{S}(bits::Union{BitVector,Vector{Bool}}) where S = FermionBasisState{S}(focknbr(bits),length(bits))
focknbr(state::FermionBasisState)  = state.focknbr
bits(state::FermionBasisState) = bits(focknbr(state),chainlength(state))
species(::FermionBasisState{S}) where S = S

@inline @generated function cellindex(::Val{S},::Val{SS}) where {S,SS}
    idx = findfirst(y->y==S,SS)
    :($idx)
end

function FermionBasisState{S}(sites,chainlength) where S
    fn = sum(ss->focknbr(ss[2],length(S),cellindex(Val(ss[1]),Val(S))),sites)
    FermionBasisState{S}(fn,chainlength)
end
FermionBasisState(sites,chainlength,::FermionBasis{S}) where S = FermionBasisState{S}(sites,chainlength)



"""
chainlength(s)

Gives the number of dots in the chain
"""
chainlength(s::FermionBasisState) = s.chainlength

function Base.pairs(s::FermionBasisState{S}) where S
    N = chainlength(s)
    M = length(S)
    b = bits(s)
    ntuple(n->S[n]=>b[n:M:N],M)
end
function Base.show(io::IO, state::FermionBasisState{S}) where {S}
    N = chainlength(state)
    println(io,"FermionBasisState{$S}")
    for (species, bits) in pairs(state)
        print(io,":",species," => ")
        print.(io,Int.(bits))
        println(io)
    end
end