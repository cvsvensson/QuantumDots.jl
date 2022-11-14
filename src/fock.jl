
abstract type AbstractBasis end
abstract type AbstractBasisState{B<:AbstractBasis} end
abstract type AbstractState{B<:AbstractBasisState} end
const DEFAULT_FERMION_SYMBOL = :f

focknbr(bits::Union{BitVector,Vector{Bool}}) = mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))
focknbr(sites::Vector{<:Integer}) = mapreduce(site -> 2^(site-1),+, sites)
focknbr(sites::Vector{<:Integer},cell_length, species_index=1) = mapreduce(site->2^(digitposition(site,cell_length,species_index)-1),+, sites)
bits(s::Integer,N) = digits(Bool,s, base=2, pad=N)

struct Fermion{ID} end
struct FermionBasis{IDs} <: AbstractBasis end
Fermion(id) = Fermion{id}()
Fermion(args...) = Fermion{args}()
FermionBasis(N::Integer) = FermionBasis{ntuple(i->(DEFAULT_FERMION_SYMBOL,i),N)}()
FermionBasis(id) = FermionBasis{id}()

FermionBasis(chainlength::Integer,species) = FermionBasis{Tuple(map(s->Symbol(s...),Base.product(species,1:chainlength)))}()
FermionBasis(chainlength::Integer,species::Symbol) = FermionBasis{ntuple(i->Symbol(species,i),chainlength)}()
nbr_of_fermions(::FermionBasis{S}) where S = length(S)
siteindex(::Fermion{ID},::FermionBasis{IDs}) where {ID,IDs} = siteindex(Val(ID),Val(IDs))
@inline @generated function siteindex(::Val{ID},::Val{IDs}) where {ID,IDs}
    idx = findfirst(y->y==ID,IDs)
    :($idx)
end

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