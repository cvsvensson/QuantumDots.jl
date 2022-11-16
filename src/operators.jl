abstract type AbstractOperator{Bin,Bout} end
Base.size(op::AbstractOperator) = (length(imagebasis(op)),length(preimagebasis(op)))
preimagebasis(::AbstractOperator{Bin}) where Bin = Bin()
imagebasis(::AbstractOperator{<:Any,Bout}) where Bout = Bout()

"""
    jwstring(site,focknbr)
    
Count the number of fermions to the right of site.
"""
jwstring(site,focknbr) = (-1)^(count_ones(focknbr >> site))

struct CreationOperator{P,Bin,Bout} <: AbstractOperator{Bin,Bout} end
Base.eltype(::CreationOperator) = Float64
CreationOperator(::P,::B) where {P<:AbstractParticle,B<:AbstractBasis} = _toLinearMap(CreationOperator{P,B,B}())
FermionCreationOperator(id::Symbol,::B) where B<:AbstractBasis = _toLinearMap(CreationOperator{Fermion{id},B,B}())
particle(::CreationOperator{P}) where P = P()

apply(op::CreationOperator,ind,basis) = addparticle(particle(op),ind,basis)
apply(op::CreationOperator,ind) = addparticle(particle(op),ind,preimagebasis(op))
function Base.:*(Cdag::CreationOperator, state) 
    out = zero(state)
    mul!(out,Cdag,state)
end
function LinearAlgebra.mul!(state2,op::AbstractOperator{Bin,Bout}, state) where {Bin,Bout}
    for (ind,val) in pairs(state)
        state_amp = apply(op, ind)
        for (basisstate,amp) in state_amp
            state2[index(basisstate,Bout())] += val*amp
        end
    end
    return state2
end
_toLinearMap(op::AbstractOperator,args...;kwargs...) = LinearMap{eltype(op)}((y,x)->mul!(y,op,x),size(op)...,args...,kwargs...)


function addfermion(digitpos::Integer,focknbr)
    cdag = 2^(digitpos-1)
    newfocknbr = cdag | focknbr
    # Check if there already was a fermion at the site. 
    allowed = iszero(cdag & focknbr) # or maybe count_ones(newfocknbr) == 1 + count_ones(focknbr)? 
    fermionstatistics = jwstring(digitpos,focknbr) #1 or -1, depending on the nbr of fermions to the right of site
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