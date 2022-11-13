
abstract type AbstractBasis end
abstract type AbstractBasisState{B<:AbstractBasis} end
abstract type AbstractState{B<:AbstractBasisState} end
const DEFAULT_FERMION_SYMBOL = :f

focknbr(bits::Union{BitVector,Vector{Bool}}) = mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))
focknbr(sites::Vector{<:Integer}) = mapreduce(site -> 2^(site-1),+, sites)
bits(s::Integer,N) = BitVector(digits(s, base=2, pad=N))


struct FermionFockBasis{S} <: AbstractBasis end
FermionFockBasis() = FermionFockBasis{(DEFAULT_FERMION_SYMBOL,)}()
#FermionFockBasis(args...) = FermionFockBasis{args}() #Should we allow this type unstable constructor?
species(::FermionFockBasis{S}) where S = S

struct Fermion{S}
    site::Int
end
species(::Fermion{S}) where S = S

struct FermionBasisState{S,NT<:NamedTuple} #How to strike a balance between lightweight and dispatch?
    spec_fock::NT
    chainlength::Int
    function FermionBasisState{S}(focknbrs::NTuple{M,Int},chainlength) where {M,S}
        @assert length(S) == M
        nt = NamedTuple{S}(focknbrs)
        new{S,typeof(nt)}(nt,chainlength)
    end
end
FermionBasisState(bits,::FermionFockBasis{S}) where S = FermionBasisState{S}(bits)
FermionBasisState(focknbr,length,::FermionFockBasis{S}) where S = FermionBasisState{S}(focknbr,length)
FermionBasisState{S}(focknbr::Integer,length::Integer) where S = FermionBasisState{(S,)}((focknbr,),length)
FermionBasisState{S}(bits::Union{BitVector,Vector{Bool}}) where S = FermionFockBasisState{S}(focknbr(bits),length(bits))
Base.getindex(state::FermionBasisState,s::Symbol) = state.spec_fock[s]
focknbr(state::FermionBasisState,species::Symbol) = state[species]
focknbr(state::FermionBasisState{S}) where S= (@assert length(S) == 1; state[S[1]])

function FermionBasisState{S}(sites::Vector{Vector{<:Integer}},chainlength) where S
    @assert length(sites) == length(S)
    @assert all(length.(sites) .<= chainlength) "Too many sites"
    @assert all(map(s-> all(s .<= chainlength),sites)) "All sites must be in the chain"
    @assert all(allunique.(sites)) "Sites must be unique"
    FermionBasisState{S}(map(focknbr,sites),length)
end
FermionBasisState{S}(sites::Vector{<:Integer},length) where S = (@assert S isa Symbol; FermionBasisState{(S,)}([sites],length))
species(::FermionBasisState{S}) where S = S

"""
    focknbr(s)

Gives the underlying number representation of the state
"""
focknbr(s::FermionBasisState{S,NamedTuple{<:Any,NTuple{1,<:Any}}}) where S = s[S[1]]
focknbrs(s::FermionBasisState{S}) where S = values(s.spec_fock)

"""
chainlength(s)

Gives the number of dots in the chain
"""
chainlength(s::FermionBasisState) = s.chainlength
Base.pairs(s::FermionBasisState) = pairs(s.spec_fock)

bits(state::FermionBasisState{S}) where S = NamedTuple{S}(bits.(focknbrs(state),chainlength(state)))

function Base.show(io::IO, state::FermionBasisState{S}) where {S}
    N = chainlength(state)
    println(io,"FermionBasisState{$S}")
    # bs = bits.(state)
    for (species, focknbr) in pairs(state)
        print(io,":",species," => ")
        print.(io,Int.(bits(focknbr,N)))
        println(io)
    end
    # print(io,:↑)
    # print.(io,b[1:N])
    # println(io)
    # print(io,:↓)
    # print.(io,b[N+1:end])
end