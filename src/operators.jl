abstract type AbstractOperatorTemplate end
abstract type AbstractOperator{Bin<:AbstractBasis,Bout<:AbstractBasis} end
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

struct CreationOperator{P} <: AbstractOperatorTemplate
    particle::P
end
Base.eltype(::CreationOperator) = Int
CreationOperator(p::P,bin::Bin,bout::Bout) where {P<:AbstractParticle,Bin<:AbstractBasis,Bout<:AbstractBasis} = Operator(CreationOperator(p),bin,bout)
CreationOperator(p::P,b::B) where {P<:AbstractParticle,B<:AbstractBasis} =CreationOperator(p,b,b)
FermionCreationOperator(id,bin::Bin,bout::Bout) where {Bin<:AbstractBasis,Bout<:AbstractBasis} = CreationOperator(Fermion(id),bin,bout)
FermionCreationOperator(id,b::B) where B<:AbstractBasis = FermionCreationOperator(id,b,b)
particle(c::CreationOperator) = c.particle


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

Base.adjoint(c::CreationOperator) = AnnihilationOperator(particle(c))

function addfermion(digitpos::Integer,focknbr)
    cdag = 2^(digitpos-1)
    newfocknbr = cdag | focknbr
    # Check if there already was a fermion at the site. 
    allowed = iszero(cdag & focknbr) # or maybe count_ones(newfocknbr) == 1 + count_ones(focknbr)? 
    fermionstatistics = jwstring(digitpos,focknbr) #1 or -1, depending on the nbr of fermions to the right of site
    return ((newfocknbr, allowed * fermionstatistics),)
end

function togglefermions(digitposvec::Vector{<:Integer}, daggers::BitVector, focknbr)
    newfocknbr = 0
    allowed = 0
    fermionstatistics = 1
    for (digitpos, dagger) in zip(digitposvec, daggers)
        op = 2^(digitpos - 1)
        if dagger
            newfocknbr = op | focknbr
            # Check if there already was a fermion at the site.
            allowed = iszero(op & focknbr)
        else
            newfocknbr = op ⊻ focknbr
            # Check if the site was empty.
            allowed = !iszero(op & focknbr)
        end
        # return directly if we create/annihilate an occupied/empty state
        if !allowed
            return ((newfocknbr, allowed * fermionstatistics),)
        end
        fermionstatistics *= jwstring(digitpos, focknbr)
        focknbr = newfocknbr
    end
    # fermionstatistics better way?
    return ((newfocknbr, allowed * fermionstatistics),)
end

# struct OperatorSum{Bo,Bi,T} <: AbstractOperator{Bo,Bi}
#     amplitudes::
#     opsum::Vector{AbstractOperator{Bo,Bi}} #Type unstable, but I'm not sure if it's desirable to be type stable here

# end
# struct OperatorProduct{Ops,Bo,Bi,T}
#     amplitude::T
#     operators::Ops
# end
