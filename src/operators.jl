Base.size(op::AbstractFockOperator) = (length(imagebasis(op)),length(preimagebasis(op)))
Base.size(op::AbstractFockOperator,k) = (length(imagebasis(op)),length(preimagebasis(op)))[k]

"""
    jwstring(site,focknbr)
    
Count the number of fermions to the right of site.
"""
jwstring(site,focknbr) = (-1)^(count_ones(focknbr >> site))

preimagebasis(op::FockOperator) = op.preimagebasis
imagebasis(op::FockOperator) = op.imagebasis
Base.eltype(op::FockOperator) = eltype(op.op)

preimagebasis(::AbstractFockOperator{Missing}) = missing
imagebasis(::AbstractFockOperator{<:Any,Missing}) = missing
Base.eltype(::CreationOperator) = Int
CreationOperator(f::Fermion) = CreationOperator((f,),(true,))
Base.:*(f1::Fermion,f2::Fermion) = CreationOperator(f1)'*CreationOperator(f2)'
Base.:*(f1::Fermion,f2::CreationOperator) = CreationOperator(f1)*f2
Base.:*(f1::CreationOperator,f2::Fermion) = f1*CreationOperator(f2)
function Base.:*(c1::CreationOperator,c2::CreationOperator)
    CreationOperator((particles(c2)...,particles(c1)...),(c2.types...,c1.types...))
end
FermionCreationOperator(id,bin::Bin,bout::Bout) where {Bin<:AbstractBasis,Bout<:AbstractBasis} = CreationOperator(Fermion(id),bin,bout)
FermionCreationOperator(id,b::B) where B<:AbstractBasis = FermionCreationOperator(id,b,b)
CreationOperator(p::P,bin::Bin,bout::Bout) where {P<:AbstractParticle,Bin<:AbstractBasis,Bout<:AbstractBasis} = FockOperator(CreationOperator(p),bin,bout)
CreationOperator(p::P,b::B) where {P<:AbstractParticle,B<:AbstractBasis} = CreationOperator(p,b,b)
particles(c::CreationOperator) = c.particles
Base.adjoint(c::CreationOperator) = CreationOperator(c.particles,broadcast(!,c.types))

apply(op::FockOperator,ind::Integer, bin = preimagebasis(op),bout=imagebasis(op)) = apply(op.op,ind,bin,bout)

siteindices(ps,bin) = map(p->siteindex(p,bin),ps)
function apply(op::CreationOperator{<:Fermion}, ind,bin, bout)
    newstate, newamp = addfermion(siteindices(particles(op),bin),basisstate(ind,bin))
    newind = index(newstate,bout)
    newind, newamp
end
index(basisstate::Integer,::FermionBasis) = basisstate+1
basisstate(ind::Integer,::FermionBasis) = ind-1

function Base.:*(op::AbstractFockOperator, state) 
    out = zero(state)
    mul!(out,op,state)
end
function LinearAlgebra.mul!(state2,op::AbstractFockOperator, state)
    state2 .*= 0
    bin = promote_basis(preimagebasis(op),basis(state))
    bout = promote_basis(imagebasis(op),basis(state2))
    for (ind,val) in pairs(state)
        newind, amp = apply(op, ind,bin,bout)
        state2[newind] += val*amp
    end
    return state2
end

function LinearAlgebra.mul!(state2,ops::FockOperatorSum, state)
    state2 .*= 0
    bin = promote_basis(preimagebasis(ops),basis(state))
    bout = promote_basis(imagebasis(ops),basis(state2))
    for (op,opamp) in pairs(ops)
        for (ind,val) in pairs(state)
            newind, amp = apply(op, ind,bin,bout)
            state2[newind] += opamp*val*amp
        end
    end
    return state2
end
LinearMaps.LinearMap(op::AbstractFockOperator,args...;kwargs...) = LinearMap{eltype(op)}((y,x)->mul!(y,op,x),(y,x)->mul!(y,op',x),size(op)...,args...,kwargs...)

preimagebasis(op::FockOperatorSum) = op.preimagebasis
imagebasis(op::FockOperatorSum) = op.imagebasis
amplitudes(op::FockOperatorSum) = op.amplitudes
operators(op::FockOperatorSum) = op.operators
Base.eltype(::FockOperatorSum{<:Any,<:Any,T}) where T = T
FockOperatorSum(op::Union{FockOperator,AbstractFockOperator}) = FockOperatorSum([one(eltype(op))],[op],preimagebasis(op),imagebasis(op))
FockOperatorSum(op::FockOperatorProduct) = FockOperatorSum([one(eltype(op))],[op],preimagebasis(op),imagebasis(op))
FockOperatorSum(op::FockOperatorSum) = op

promote_basis(b::AbstractBasis,::Missing) = b
promote_basis(::Missing,b::AbstractBasis) = b
promote_basis(::Missing,b::Missing) = missing
promote_basis(b1::B,b2::B) where B<:AbstractBasis = (@assert b1==b2 "Basis must match"; b1)

Base.:+(o1::Union{FockOperator,FockOperatorProduct,FockOperatorSum},o2::Union{FockOperator,FockOperatorProduct,FockOperatorSum}) = FockOperatorSum(o1) + FockOperatorSum(o2)
Base.:-(o1::Union{FockOperator,FockOperatorProduct,FockOperatorSum},o2::Union{FockOperator,FockOperatorProduct,FockOperatorSum}) = FockOperatorSum(o1) + (-FockOperatorSum(o2))
function Base.:+(opsum1::FockOperatorSum,opsum2::FockOperatorSum)
    newamps = vcat(amplitudes(opsum1),amplitudes(opsum2))
    newops = vcat(operators(opsum1),operators(opsum2))
    newpreimagebasis = promote_basis(preimagebasis(opsum1),preimagebasis(opsum1))
    newimagebasis = promote_basis(imagebasis(opsum1),imagebasis(opsum1))
    FockOperatorSum(newamps,newops,newpreimagebasis,newimagebasis)
end
Base.:-(opsum::FockOperatorSum) = FockOperatorSum(-amplitudes(opsum),operators(opsum),preimagebasis(opsum),imagebasis(opsum))
Base.:-(op::FockOperator) = -FockOperatorSum(op)

preimagebasis(op::FockOperatorProduct) = op.preimagebasis
imagebasis(op::FockOperatorProduct) = op.imagebasis
amplitude(op::FockOperatorProduct) = op.amplitude
operators(op::FockOperatorProduct) = op.operators
Base.eltype(op::FockOperatorProduct) = promote_type(eltype.(operators(op))...)
FockOperatorProduct(op::FockOperator) = FockOperatorProduct((op,),preimagebasis(op),imagebasis(op))
FockOperatorProduct(op::CreationOperator) = FockOperatorProduct((op,),preimagebasis(op),imagebasis(op))

Base.:*(x::Number,op::AbstractFockOperator) = x*FockOperatorSum(op)
Base.:*(op::FockOperator,x::Number) = x*op
Base.:*(x::Number,op::FockOperator) = x*FockOperatorSum(op)
Base.:*(x::Number,op::FockOperatorProduct) = x*FockOperatorSum(op)
Base.:*(x::Number,op::FockOperatorSum) = FockOperatorSum(x.*amplitudes(op),operators(op),preimagebasis(op),imagebasis(op))
Base.:*(op1::FockOperator,op2::FockOperator) = FockOperatorProduct(op1) * FockOperatorProduct(op2)
Base.:*(op1::FockOperatorProduct,op2::FockOperator) = op1 * FockOperatorProduct(op2)
Base.:*(op1::FockOperator,op2::FockOperatorProduct) = FockOperatorProduct(op1) * op2
function Base.:*(op1::FockOperatorProduct,op2::FockOperatorProduct) 
    newops = (operators(op1)...,operators(op2)...)
    FockOperatorProduct(newops,preimagebasis(op2),imagebasis(op1))
end
Base.:*(op1::AbstractFockOperator,op2::AbstractFockOperator) = FockOperatorProduct(op1)*FockOperatorProduct(op2)
function Base.:*(op1::FockOperatorSum,op2::FockOperatorSum)
    newops = vec(map(ops->ops[1]*ops[2],Base.product(operators(op1),operators(op2))))
    newamps = vec(map(amps->amps[1]*amps[2],Base.product(amplitudes(op1),amplitudes(op2))))
    FockOperatorSum(newamps,newops,preimagebasis(op2),imagebasis(op1))
end

function keysvalues(d)
    k = collect(keys(d)) 
    v = map(key->d[key],k) 
    return k,v
end
function groupbykeysandreduce(k::K,v::V,f) where {K,V}
    d = groupreduce(first,last,f,zip(k,v))
    ks::K = collect(keys(d))
    vs::V = collect(values(d))
    return ks, vs
end
function groupoperators(ops,amps)
    d = groupreduce(operators ∘ first,last,zip(ops,amps))
    ks::K = collect(keys(d))
    vs::V = collect(values(d))
    return ks, vs
end

# function apply(opsum::FockOperatorSum,ind::Integer,bin = preimagebasis(op),bout = imagebasis(op))
#     allinds = typeof(ind)[]
#     allamps = eltype(opsum)[]
#     for (op,scaling) in pairs(opsum)
#         newinds, amps = apply(op,ind,bin,bout)
#         newamps = amps .* (scaling)
#         allinds = vcat(allinds,newinds)
#         allamps = vcat(allamps,newamps)
#     end
#     return groupbykeysandreduce(allinds,allamps,+)
# end
Base.pairs(opsum::FockOperatorSum) = zip(operators(opsum),amplitudes(opsum))

function apply(op::FockOperatorProduct,ind,bin = preimagebasis(op),bout = imagebasis(op))
    function _apply(op,ind,scale)
        bin = promote_basis(preimagebasis(op), bin)
        bin = promote_basis(imagebasis(op), bout)
        newind,newamp = apply(op,ind,bin,bout)
        newind, scale*newamp
    end
    foldr((op,ia) -> _apply(op,ia...),operators(op),init=(ind,one(eltype(op))))
end