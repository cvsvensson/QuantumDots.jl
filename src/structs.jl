abstract type AbstractBasis end
# abstract type AbstractOperator{Bin<:Union{AbstractBasis,Missing},Bout<:Union{AbstractBasis,Missing}} end
abstract type AbstractFockOperator{Bin<:Union{AbstractBasis,Missing},Bout<:Union{AbstractBasis,Missing}} end
abstract type AbstractParticle end
const DEFAULT_FERMION_SYMBOL = :f
const BasisOrMissing = Union{AbstractBasis,Missing}
basis(::AbstractArray) = missing

struct Fermion{S} <: AbstractParticle 
    id::S
end
struct FermionBasis{M,S} <: AbstractBasis
    ids::NTuple{M,S}
end

struct CreationOperator{P,M} <: AbstractFockOperator{Missing,Missing}
    particles::NTuple{M,P}
    types::NTuple{M,Bool} # true is creation, false is annihilation
end

struct FockOperator{Bin,Bout,Op} <: AbstractFockOperator{Bin,Bout} 
    op::Op
    preimagebasis::Bin
    imagebasis::Bout
end

struct FockOperatorSum{Bin,Bout,T,Ops} <: AbstractFockOperator{Bin,Bout}
    amplitudes::Vector{T}
    operators::Vector{Ops}
    preimagebasis::Bin
    imagebasis::Bout
    function FockOperatorSum(amplitudes::Vector{T},ops::Vector{Ops},bin::Bin,bout::Bout) where {Ops,T,Bin<:BasisOrMissing,Bout<:BasisOrMissing}
        newops, newamps = groupbykeysandreduce(ops,amplitudes,+)
        new{Bin,Bout,promote_type(T,eltype.(ops)...),Ops}(newamps,newops,bin,bout)
    end
end
