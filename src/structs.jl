abstract type AbstractBasis end
# abstract type AbstractOperator{Bin<:Union{AbstractBasis,Missing},Bout<:Union{AbstractBasis,Missing}} end
abstract type AbstractFockOperator{Bin<:Union{AbstractBasis,Missing},Bout<:Union{AbstractBasis,Missing}} end
abstract type AbstractElementaryFockOperator{Bin,Bout} <: AbstractFockOperator{Bin,Bout} end
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

struct CreationOperator{P,M} <: AbstractElementaryFockOperator{Missing,Missing}
    particles::NTuple{M,P}
    types::NTuple{M,Bool} # true is creation, false is annihilation
end

struct FockOperator{Bin,Bout,Op<:AbstractElementaryFockOperator} <: AbstractFockOperator{Bin,Bout} 
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

struct FockOperatorProduct{Ops,Bin,Bout} <: AbstractFockOperator{Bin,Bout}
    operators::Ops
    preimagebasis::Bin
    imagebasis::Bout
    function FockOperatorProduct(ops::Ops,bin::Bin,bout::Bout) where {Ops,Bin<:BasisOrMissing,Bout<:BasisOrMissing}
        new{Ops,Bin,Bout}(ops,bin,bout)
    end
end
preimagebasis(op::FockOperatorProduct) = op.preimagebasis
imagebasis(op::FockOperatorProduct) = op.imagebasis
amplitude(op::FockOperatorProduct) = op.amplitude
operators(op::FockOperatorProduct) = op.operators
Base.eltype(op::FockOperatorProduct) = promote_type(eltype.(operators(op))...)
FockOperatorProduct(op::AbstractFockOperator) = FockOperatorProduct((op,),preimagebasis(op),imagebasis(op))
# FockOperatorProduct(op::CreationOperator) = FockOperatorProduct((op,),preimagebasis(op),imagebasis(op))


function apply(op::FockOperatorProduct,ind,bin = preimagebasis(op),bout = imagebasis(op))
    function _apply(op,ind,scale)
        bin = promote_basis(preimagebasis(op), bin)
        bout = promote_basis(imagebasis(op), bout)
        newind, newamp = apply(op,ind,bin,bout)
        newind, scale*newamp
    end
    foldr((op,ia) -> _apply(op,ia...),operators(op),init=(ind,one(eltype(op))))
end

Base.:*(x::Number,op::FockOperatorProduct) = x*FockOperatorSum(op)

Base.:+(o1::Union{FockOperator,FockOperatorProduct,FockOperatorSum},o2::Union{FockOperator,FockOperatorProduct,FockOperatorSum}) = FockOperatorSum(o1) + FockOperatorSum(o2)
Base.:-(o1::Union{FockOperator,FockOperatorProduct,FockOperatorSum},o2::Union{FockOperator,FockOperatorProduct,FockOperatorSum}) = FockOperatorSum(o1) + (-FockOperatorSum(o2))