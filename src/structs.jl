abstract type AbstractBasis end
# abstract type AbstractOperator{Bin<:Union{AbstractBasis,Missing},Bout<:Union{AbstractBasis,Missing}} end
abstract type AbstractFockOperator{Bin<:Union{AbstractBasis,Missing},Bout<:Union{AbstractBasis,Missing}} end
abstract type AbstractParticle end
const DEFAULT_FERMION_SYMBOL = :f


struct Fermion{S} <: AbstractParticle 
    id::S
end
struct FermionBasis{M,S} <: AbstractBasis
    ids::NTuple{M,S}
end

const BasisOrMissing = Union{AbstractBasis,Missing}
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
struct FockOperatorProduct{Ops,Bin,Bout} <: AbstractFockOperator{Bin,Bout}
    operators::Ops
    preimagebasis::Bin
    imagebasis::Bout
    function FockOperatorProduct(ops::Ops,bin::Bin,bout::Bout) where {Ops,Bin<:BasisOrMissing,Bout<:BasisOrMissing}
        new{Ops,Bin,Bout}(ops,bin,bout)
    end
end