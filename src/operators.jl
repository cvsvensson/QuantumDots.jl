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
function Base.:*(c1::FockOperator{<:Any,<:Any,<:CreationOperator},c2::FockOperator{<:Any,<:Any,<:CreationOperator})
    FockOperator(c1.op*c2.op,preimagebasis(c2),imagebasis(c1))
end
Base.:*(c1::CreationOperator,c2::FockOperator{<:Any,<:Any,<:CreationOperator}) = FockOperator(c1*c2.op,preimagebasis(c2),imagebasis(c2))
Base.:*(c1::FockOperator{<:Any,<:Any,<:CreationOperator},c2::CreationOperator) = FockOperator(c1.op*c2,preimagebasis(c1),imagebasis(c1))

FermionCreationOperator(id,bin::Bin,bout::Bout) where {Bin<:AbstractBasis,Bout<:AbstractBasis} = CreationOperator(Fermion(id),bin,bout)
FermionCreationOperator(id,b::B) where B<:AbstractBasis = FermionCreationOperator(id,b,b)
CreationOperator(p::P,bin::Bin,bout::Bout) where {P<:AbstractParticle,Bin<:AbstractBasis,Bout<:AbstractBasis} = FockOperator(CreationOperator(p),bin,bout)
CreationOperator(p::P,b::B) where {P<:AbstractParticle,B<:AbstractBasis} = CreationOperator(p,b,b)
particles(c::CreationOperator) = c.particles
Base.adjoint(c::CreationOperator) = CreationOperator(c.particles,broadcast(!,c.types))

apply(op::FockOperator,ind::Integer, bin = preimagebasis(op),bout=imagebasis(op)) = apply(op.op,ind,bin,bout)

siteindices(ps,bin) = map(p->siteindex(p,bin),ps)
function apply(op::CreationOperator{<:Fermion}, ind,bin, bout)
    newstate, newamp = togglefermions(siteindices(particles(op),bin),op.types,basisstate(ind,bin))
    newind = index(newstate,bout)
    newind, newamp
end
index(basisstate::Integer,::FermionBasis) = basisstate+1
basisstate(ind::Integer,::FermionBasis) = ind-1

function Base.:*(op::AbstractFockOperator, state::AbstractVector)
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
FockOperatorSum(op::FockOperatorSum) = op

promote_basis(b::AbstractBasis,::Missing) = b
promote_basis(::Missing,b::AbstractBasis) = b
promote_basis(::Missing,b::Missing) = missing
promote_basis(b1::B,b2::B) where B<:AbstractBasis = (@assert b1==b2 "Basis must match"; b1)

Base.:+(o1::Union{FockOperator,FockOperatorSum},o2::Union{FockOperator,FockOperatorSum}) = FockOperatorSum(o1) + FockOperatorSum(o2)
Base.:-(o1::Union{FockOperator,FockOperatorSum},o2::Union{FockOperator,FockOperatorSum}) = FockOperatorSum(o1) + (-FockOperatorSum(o2))
function Base.:+(opsum1::FockOperatorSum,opsum2::FockOperatorSum)
    newamps = vcat(amplitudes(opsum1),amplitudes(opsum2))
    newops = vcat(operators(opsum1),operators(opsum2))
    newpreimagebasis = promote_basis(preimagebasis(opsum1),preimagebasis(opsum1))
    newimagebasis = promote_basis(imagebasis(opsum1),imagebasis(opsum1))
    FockOperatorSum(newamps,newops,newpreimagebasis,newimagebasis)
end
Base.:-(opsum::FockOperatorSum) = FockOperatorSum(-amplitudes(opsum),operators(opsum),preimagebasis(opsum),imagebasis(opsum))
Base.:-(op::FockOperator) = -FockOperatorSum(op)


function togglefermions(digitpositions, daggers, focknbr)
    newfocknbr = 0
    allowed = 0
    fermionstatistics = 1
    for (digitpos, dagger) in zip(digitpositions, daggers)
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
            return newfocknbr, allowed * fermionstatistics
        end
        fermionstatistics *= jwstring(digitpos, focknbr)
        focknbr = newfocknbr
    end
    # fermionstatistics better way?
    return newfocknbr, allowed * fermionstatistics
end

Base.:*(x::Number,op::AbstractFockOperator) = x*FockOperatorSum(op)
Base.:*(op::FockOperator,x::Number) = x*op
Base.:*(x::Number,op::FockOperator) = x*FockOperatorSum(op)
Base.:*(x::Number,op::FockOperatorSum) = FockOperatorSum(x.*amplitudes(op),operators(op),preimagebasis(op),imagebasis(op))
function Base.:*(op1::FockOperatorSum,op2::FockOperatorSum)
    newops = vec(map(ops->ops[1]*ops[2],Base.product(operators(op1),operators(op2))))
    newamps = vec(map(amps->amps[1]*amps[2],Base.product(amplitudes(op1),amplitudes(op2))))
    FockOperatorSum(newamps,newops,preimagebasis(op2),imagebasis(op1))
end

function groupbykeysandreduce(k::K,v::V,f) where {K,V}
    d = groupreduce(first,last,f,zip(k,v))
    ks::K = collect(keys(d))
    vs::V = collect(values(d))
    return ks, vs
end

Base.pairs(opsum::FockOperatorSum) = zip(operators(opsum),amplitudes(opsum))
