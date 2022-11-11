
abstract type AbstractBasis end
abstract type AbstractBasisState{B<:AbstractBasis} end
abstract type AbstractState{B<:AbstractBasisState} end
struct SpinlessFockBasis{N} <: AbstractBasis end
struct SpinHalfFockBasis{N} <: AbstractBasis end

struct SpinlessFockBasisState{N,T} <: AbstractBasisState{SpinlessFockBasis{N}}
    state::T
    function SpinlessFockBasisState{N}(num::T) where {N,T}
        new{N,T}(num)
    end
end
SpinlessFockBasisState(bits::Union{BitVector,Vector{Bool}}) = SpinlessFockBasisState{length(bits)}(mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits)))
SpinlessFockBasisState{N}(bits::Union{BitVector,Vector{Bool}}) where N = (@assert length(bits) == N;SpinlessFockBasisState{N}(mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))))
function SpinlessFockBasisState{N}(sites::Vector{Integer}) where N 
    @assert length(sites) <= N
    @assert length(unique(sites)) == length(sites)
    SpinlessFockBasisState{N}(mapreduce(site -> 2^(site-1),+, sites))
end
SpinlessFockBasisState{N}(;site::Integer) where N = SpinlessFockBasisState{N}(2^(site-1))

struct SpinHalfFockBasisState{N,T} <: AbstractBasisState{SpinHalfFockBasis{N}}
    state::T
    function SpinHalfFockBasisState{N}(num::T) where {N,T}
        new{N,T}(num)
    end
end

SpinHalfFockBasisState(bits::Union{BitVector,Vector{Bool}}) = SpinHalfFockBasisState{length(bits) ÷ 2}(mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits)))
SpinHalfFockBasisState{N}(bits::Union{BitVector,Vector{Bool}}) where N = (@assert length(bits) == 2N;SpinHalfFockBasisState{N}(mapreduce(nb -> nb[2] * 2^(nb[1]-1),+, enumerate(bits))))
function SpinHalfFockBasisState{N}(upsites::Vector{Integer},dnsites::Vector{Integer}) where N 
    SpinHalfFockBasisState{N}(SpinlessFockBasisState{N}(upsites),SpinlessFockBasisState{N}(dnsites))
end

"""
    SpinHalfFockBasisState{N}(fup,fdn)

Construct a Fock basis state from fup and fdn
"""
SpinHalfFockBasisState{N}(fup::T,fdn::T) where {N,T<:Number} = SpinHalfFockBasisState{N}(fup + (fdn << N))
function SpinHalfFockBasisState(fup::SpinlessFockBasisState{N},fdn::SpinlessFockBasisState{N}) where N
    SpinHalfFockBasisState{N}(focknbr(fup) + (focknbr(fdn) << N))
end

struct State{N,BS<:AbstractBasisState,T} <: AbstractState{BS}
    basisstates::Vector{BS}
    amplitudes::Vector{T}
end

"""
    focknbr(s)

Gives the underlying number representation of the state
"""
focknbr(s::Union{SpinHalfFockBasisState,SpinlessFockBasisState}) = s.state

"""
chainlength(s)

Gives the number of dots in the chain
"""
chainlength(::Union{SpinHalfFockBasisState{N},SpinlessFockBasisState{N}}) where N = N
chainlength(::Union{SpinHalfFockBasis{N},SpinlessFockBasis{N}}) where N = N

bits(s::Integer,N) = digits(s, base=2, pad=N)
bits(::SpinlessFockBasisState{<:Any,Missing}) = missing
bits(::SpinHalfFockBasisState{<:Any,Missing}) = missing
bits(state::SpinlessFockBasisState{N}) where N = digits(focknbr(state), base=2, pad=N)
bits(state::SpinHalfFockBasisState{N}) where N = digits(focknbr(state), base=2, pad=2*N)
Base.show(io::IO, state::SpinlessFockBasisState{N,T}) where {N,T} = (println("SpinlessFockBasisState{$N,$T}");print.(io,bits(state))) #for b in bits(state) print(io, b) end
function Base.show(io::IO, state::SpinHalfFockBasisState{N,T}) where {N,T}
    println("SpinHalfFockBasisState{$N,$T}")
    b = bits(state)
    print(io,:↑)
    print.(io,b[1:N])
    println(io)
    print(io,:↓)
    print.(io,b[N+1:end])
end