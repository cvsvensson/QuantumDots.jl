
abstract type AbstractBasis end
abstract type AbstractParticle end

const DEFAULT_FERMION_SYMBOL = :f

focknbr(bits::Union{BitVector,Vector{Bool}}) = mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))
focknbr(sites::Vector{<:Integer}) = mapreduce(site -> 2^(site-1),+, sites)
focknbr(sites::Vector{<:Integer},cell_length, species_index=1) = mapreduce(site->2^(digitposition(site,cell_length,species_index)-1),+, sites)
bits(s::Integer,N) = digits(Bool,s, base=2, pad=N)

struct Fermion{S} <: AbstractParticle 
    id::S
end
Base.adjoint(f::Fermion) 
struct FermionBasis{M,S} <: AbstractBasis
    ids::NTuple{M,S}
    # FermionBasis(ids::NTuple{M,S}) where {M,S} = new{M,S}(ids)
end
Fermion(args...) = Fermion(args)
FermionBasis(N::Integer) = FermionBasis(ntuple(i->(DEFAULT_FERMION_SYMBOL,i),N))
particles(b::FermionBasis) = Fermion.(b.ids)

FermionBasis(chainlength::Integer,species) = FermionBasis(Tuple(Base.product(species,1:chainlength)))
FermionBasis(chainlength::Integer,species::Symbol) = FermionBasis(ntuple(i->(species,i),chainlength))
nbr_of_fermions(::FermionBasis{M}) where M = M
Base.length(b::FermionBasis) = 2^nbr_of_fermions(b)
siteindex(f::Fermion,b::FermionBasis) = findfirst(x->x==f.id,b.ids)
# siteindex(f::Fermion,b::FermionBasis) = siteindex(Val(f.id),Val(b.ids))
# @inline @generated function siteindex(::Val{ID},::Val{IDs}) where {ID,IDs}
#     idx = findfirst(y->y==ID,IDs)
#     :($idx)
# end

# function Base.pairs(s::FermionBasisState{S}) where S
#     # N = chainlength(s)
#     M = length(S)
#     b = bits(s)
#     ntuple(n->S[n]=>b[n:M:end],M)
# end
# function Base.show(io::IO, state::FermionBasisState{S}) where {S}
#     println(io,"FermionBasisState{$S}")
#     for (species, bits) in pairs(state)
#         print(io,":",species," => ")
#         print.(io,Int.(bits))
#         println(io)
#     end
# end