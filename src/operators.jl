# abstract type AbstractOperatorTemplate end
abstract type AbstractOperator{Bin<:Union{AbstractBasis,Missing},Bout<:Union{AbstractBasis,Missing}} end
Base.size(op::AbstractOperator) = (length(imagebasis(op)),length(preimagebasis(op)))
Base.size(op::AbstractOperator,k) = (length(imagebasis(op)),length(preimagebasis(op)))[k]


"""
    jwstring(site,focknbr)
    
Count the number of fermions to the right of site.
"""
jwstring(site,focknbr) = (-1)^(count_ones(focknbr >> site))

struct Operator{Bin,Bout,Op} <: AbstractOperator{Bin,Bout} 
    op::Op
    preimagebasis::Bin
    imagebasis::Bout
end
preimagebasis(op::Operator) = op.preimagebasis
imagebasis(op::Operator) = op.imagebasis
Base.eltype(op::Operator) = eltype(op.op)

struct CreationOperator{P} <: AbstractOperator{Missing,Missing}
    particle::P
end
Base.eltype(::CreationOperator) = Int
CreationOperator(p::P,bin::Bin,bout::Bout) where {P<:AbstractParticle,Bin<:AbstractBasis,Bout<:AbstractBasis} = Operator(CreationOperator(p),bin,bout)
CreationOperator(p::P,b::B) where {P<:AbstractParticle,B<:AbstractBasis} =CreationOperator(p,b,b)
FermionCreationOperator(id,bin::Bin,bout::Bout) where {Bin<:AbstractBasis,Bout<:AbstractBasis} = CreationOperator(Fermion(id),bin,bout)
FermionCreationOperator(id,b::B) where B<:AbstractBasis = FermionCreationOperator(id,b,b)
particle(c::CreationOperator) = c.particle
Base.adjoint(c::CreationOperator) = AnnihilationOperator(particle(c))


apply(op::Operator,ind::Integer) = apply(op.op,ind,preimagebasis(op),imagebasis(op))
apply(op::CreationOperator,ind,bin::B,bout::B) where B<:AbstractBasis = addparticle(particle(op),ind,bin,bout)

addparticle(f::Fermion, ind,bin,bout) = addparticle(f,ind,bin)
addparticle(f::Fermion, ind,bin) = addfermion(siteindex(f,bin), basisstate(ind,bin))
index(basisstate::Integer,::FermionBasis) = basisstate+1
basisstate(ind::Integer,::FermionBasis) = ind-1

# apply(op::Operator,ind::Integer) = addparticle(particle(op),ind,basis)
# apply(op::CreationOperator,ind) = addparticle(particle(op),ind,preimagebasis(op))
function Base.:*(op::AbstractOperator, state) 
    out = zero(state)
    mul!(out,op,state)
end
function LinearAlgebra.mul!(state2,op::AbstractOperator, state)
    state2 .*= 0
    for (ind,val) in pairs(state)
        state_amp = apply(op, ind)
        for (state,amp) in state_amp
            state2[index(state,imagebasis(op))] += val*amp
        end
    end
    return state2
end
LinearMaps.LinearMap(op::AbstractOperator,args...;kwargs...) = LinearMap{eltype(op)}((y,x)->mul!(y,op,x),(y,x)->mul!(y,op',x),size(op)...,args...,kwargs...)

function addfermion(digitpos::Integer,focknbr)
    cdag = 2^(digitpos-1)
    newfocknbr = cdag | focknbr
    # Check if there already was a fermion at the site. 
    allowed = iszero(cdag & focknbr) # or maybe count_ones(newfocknbr) == 1 + count_ones(focknbr)? 
    fermionstatistics = jwstring(digitpos,focknbr) #1 or -1, depending on the nbr of fermions to the right of site
    return ((newfocknbr, allowed * fermionstatistics),)
end
const BasisOrMissing = Union{AbstractBasis,Missing}
struct OperatorSum{Bin,Bout,T,Ops} <: AbstractOperator{Bin,Bout}
    amplitudes::Vector{T}
    operators::Vector{Ops}
    preimagebasis::Bin
    imagebasis::Bout
    function OperatorSum(amplitudes::Vector{T},ops::Vector{Ops},bin::Bin,bout::Bout) where {Ops,T,Bin<:BasisOrMissing,Bout<:BasisOrMissing}
        new{Bin,Bout,promote_type(T,eltype.(ops)...),Ops}(amplitudes,ops,bin,bout)
    end
end

struct OperatorProduct{Ops,Bin,Bout,T} <: AbstractOperator{Bin,Bout}
    amplitude::T
    operators::Ops
    preimagebasis::Bin
    imagebasis::Bout
    function OperatorProduct(amplitude::T,ops::Ops,bin::Bin,bout::Bout) where {T,Ops,Bin<:BasisOrMissing,Bout<:BasisOrMissing}
        new{Ops,Bin,Bout,promote_type(T,eltype.(ops)...)}(amplitude,ops,bin,bout)
    end
end

preimagebasis(op::OperatorSum) = op.preimagebasis
imagebasis(op::OperatorSum) = op.imagebasis
amplitudes(op::OperatorSum) = op.amplitudes
operators(op::OperatorSum) = op.operators
Base.eltype(::OperatorSum{<:Any,<:Any,T}) where T = T
OperatorSum(op::Operator) = OperatorSum([one(eltype(op))],[op],preimagebasis(op),imagebasis(op))
OperatorSum(op::OperatorProduct) = OperatorSum([one(eltype(op))],[op],preimagebasis(op),imagebasis(op))
OperatorSum(op::OperatorSum) = op

promote_basis(b::AbstractBasis,::Missing) = b
promote_basis(::Missing,b::AbstractBasis) = b
promote_basis(::Missing,b::Missing) = missing
promote_basis(b1::B,b2::B) where B<:AbstractBasis = (@assert b1==b2 "Basis must match"; b1)

Base.:+(o1::Union{Operator,OperatorProduct,OperatorSum},o2::Union{Operator,OperatorProduct,OperatorSum}) = OperatorSum(o1) + OperatorSum(o2)
function Base.:+(opsum1::OperatorSum,opsum2::OperatorSum)
    newamps = vcat(amplitudes(opsum1),amplitudes(opsum2))
    newops = vcat(operators(opsum1),operators(opsum2))
    newpreimagebasis = promote_basis(preimagebasis(opsum1),preimagebasis(opsum1))
    newimagebasis = promote_basis(imagebasis(opsum1),imagebasis(opsum1))
    #TODO: check if any operator matches another, then we can combine terms.
    OperatorSum(newamps,newops,newpreimagebasis,newimagebasis)
end

preimagebasis(op::OperatorProduct) = op.preimagebasis
imagebasis(op::OperatorProduct) = op.imagebasis
amplitude(op::OperatorProduct) = op.amplitude
operators(op::OperatorProduct) = op.operators
Base.eltype(::OperatorProduct{<:Any,<:Any,<:Any,T}) where T = T
OperatorProduct(op::Operator) = OperatorProduct(one(eltype(op)),(op,),preimagebasis(op),imagebasis(op))

Base.:*(op::Operator,x::Number) = x*op
Base.:*(x::Number,op::Operator) = x*OperatorProduct(op)
Base.:*(x::Number,op::OperatorProduct) = OperatorProduct(x*amplitude(op),operators(op),preimagebasis(op),imagebasis(op))
Base.:*(op1::Operator,op2::Operator) = OperatorProduct(op1) * OperatorProduct(op2)
Base.:*(op1::OperatorProduct,op2::Operator) = op1 * OperatorProduct(op2)
Base.:*(op1::Operator,op2::OperatorProduct) = OperatorProduct(op1) * op2
function Base.:*(op1::OperatorProduct,op2::OperatorProduct) 
    newamp = amplitude(op1) * amplitude(op2)
    newops = (operators(op1)...,operators(op2))
    OperatorProduct(newamp,newops,preimagebasis(op2),imagebasis(op1))
end