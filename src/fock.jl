
abstract type AbstractBasis end
abstract type AbstractBasisState{B<:AbstractBasis} end
abstract type AbstractState{B<:AbstractBasisState} end
struct FermionFockBasis <: AbstractBasis 
    species::Symbol
end
const DEFAULT_FERMION_SYMBOL = :f
FermionFockBasis() = FermionFockBasis(DEFAULT_FERMION_SYMBOL)
struct SpinHalfFockBasis <: AbstractBasis end
struct ManyFermionsFockBasis{M} <: AbstractBasis 
    species::NTuple{M,Symbol}
    function ManyFermionsFockBasis(species::Vararg{Symbol,M}) where M
        new{M}(species)
    end
end
species(B::FermionFockBasis) = B.species
species(B::ManyFermionsFockBasis) = B.species

struct FermionFockBasisState{T,TA} <: AbstractBasisState{FermionFockBasis}
    state::T
    amplitude::TA
    chainlength::Int
    basis::FermionFockBasis
    function FermionFockBasisState(num::T,amp::TA,length,basis::FermionFockBasis) where {T,TA}
        new{T,TA}(num,amp,length,basis)
    end
end
FermionFockBasisState(num::Number,length) = FermionFockBasisState(num,1,length,FermionFockBasis())
FermionFockBasisState(num::Number,length,basis::FermionFockBasis) = FermionFockBasisState(num,1,length,basis)
FermionFockBasisState(bits::Union{BitVector,Vector{Bool}}) = FermionFockBasisState(focknbr(bits),length(bits))
function FermionFockBasisState(sites::Vector{<:Integer},length)
    @assert length(sites) <= length "Too many sites"
    @assert all(sites .<= length) "All sites must be in the chain"
    @assert allunique(sites) "Sites must be unique"
    FermionFockBasisState(mapreduce(site -> 2^(site-1),+, sites),length)
end
FermionFockBasisState(;site::Integer,length::Integer) = FermionFockBasisState(2^(site-1),length)
Base.:*(x::T,ψ::FermionFockBasisState) where {T} = FermionFockBasisState(focknbr(ψ),x*amplitude(ψ),chainlength(ψ),basis(ψ))
species(s::FermionFockBasisState) = species(basis(s))


struct ManyFermionsFockBasisState{M,T,TA} <: AbstractBasisState{ManyFermionsFockBasis{M}}
    states::NTuple{M,T}
    amplitude::TA
    chainlength::Int
    basis::ManyFermionsFockBasis{M}
    function ManyFermionsFockBasisState(num::NTuple{M,T},amp::TA,length,basis::ManyFermionsFockBasis{M}) where {M,T,TA}
        new{M,T,TA}(num,amp,length,basis)
    end
end
ManyFermionsFockBasisState(num::NTuple{M,T},length) where {M,T} = ManyFermionsFockBasisState(num,1,length, ManyFermionsFockBasis{M}())
ManyFermionsFockBasisState(num::NTuple{M,<:Number},length,basis::ManyFermionsFockBasis{M}) where {M} = ManyFermionsFockBasisState(num,1,length, basis)
ManyFermionsFockBasisState(num::NTuple{M,<:Union{BitVector,Vector{Bool}}},length,basis::ManyFermionsFockBasis{M}) where {M} = ManyFermionsFockBasisState(focknbr.(num),1,length, basis)

amplitude(s::Union{ManyFermionsFockBasisState,FermionFockBasisState}) = s.amplitude
basis(s::Union{ManyFermionsFockBasisState,FermionFockBasisState}) = s.basis


focknbr(bits::Union{BitVector,Vector{Bool}}) = mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))
function ManyFermionsFockBasisState(bits::NTuple{M,Union{BitVector,Vector{Bool}}}) where M
    lengths = length.(bits)
    ul = unique(lengths)
    @assert length(ul) == 1
    ManyFermionsFockBasisState(ntuple(i->focknbr(bits[i]),Val(M)),ul[1])
end
function ManyFermionsFockBasisState(upsites::Vector{Integer},dnsites::Vector{Integer})
    ManyFermionsFockBasisState(FermionFockBasisState(upsites,length),FermionFockBasisState(dnsites,length))
end

"""
    SpinHalfFockBasisState(fup,fdn)

Construct a Fock basis state from fup and fdn
"""
function ManyFermionsFockBasisState(states::Vararg{FermionFockBasisState,M}) where M
    @assert allequal(chainlength.(states))
    @assert allunique(species.(states))
    ManyFermionsFockBasisState(focknbr.(states),chainlength(states[1]))
end

struct State{N,BS<:AbstractBasisState,T} <: AbstractState{BS}
    basisstates::Vector{BS}
    amplitudes::Vector{T}
end

"""
    focknbr(s)

Gives the underlying number representation of the state
"""
focknbr(s::Union{ManyFermionsFockBasisState,FermionFockBasisState}) = s.state

"""
chainlength(s)

Gives the number of dots in the chain
"""
chainlength(s::Union{ManyFermionsFockBasisState,FermionFockBasisState}) = s.chainlength

bits(s::Integer,N) = digits(s, base=2, pad=N)
bits(::FermionFockBasisState{<:Any,Missing}) = missing
bits(::ManyFermionsFockBasisState{<:Any,Missing}) = missing
bits(state::FermionFockBasisState) = digits(focknbr(state), base=2, pad=chainlength(state))
bits(state::ManyFermionsFockBasisState) = digits.(focknbr.(state), base=2, pad=chainlength(state))
function Base.show(io::IO, state::FermionFockBasisState{T,TA}) where {T,TA} 
    print(io,amplitude(state))
    print(io,"*")
    println(io,"FermionFockBasisState{$T,$TA}")
    print(io,species(state))
    print.(io,bits(state))
    #for b in bits(state) print(io, b) end
end
states(s::ManyFermionsFockBasisState) = s.states
species(s::ManyFermionsFockBasisState) = species(basis(s))
Base.pairs(s::ManyFermionsFockBasisState) = zip(species(s),states(s))
function Base.show(io::IO, state::ManyFermionsFockBasisState{M,T,TA}) where {M,TA,T}
    N = chainlength(state)
    print(io,amplitude(state))
    print(io,"*")
    println(io,"ManyFermionsFockBasisState{$M,$T,$TA}")
    # bs = bits.(state)
    for (species, focknbr) in pairs(state)
        print(io,species)
        print.(io,bits(focknbr,N))
        println(io)
    end
    # print(io,:↑)
    # print.(io,b[1:N])
    # println(io)
    # print(io,:↓)
    # print.(io,b[N+1:end])
end