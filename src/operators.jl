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

struct AnnihilationOperator{P,M} <: AbstractOperator{Missing,Missing}
    particles::NTuple{M,P}
    types::NTuple{M,Bool} # true is creation, false is annihilation
end
Base.eltype(::AnnihilationOperator) = Int
AnnihilationOperator(f::Fermion) = AnnihilationOperator((f,),(false,))
function Base.:*(c1::AnnihilationOperator,c2::AnnihilationOperator)
    AnnihilationOperator((particles(c1)...,particles(c2)...),(c1.types...,c2.types...))
end
FermionAnnihilationOperator(id,bin::Bin,bout::Bout) where {Bin<:AbstractBasis,Bout<:AbstractBasis} = AnnihilationOperator(Fermion(id),bin,bout)
FermionAnnihilationOperator(id,b::B) where B<:AbstractBasis = FermionCreationOperator(id,b,b)
AnnihilationOperator(p::P,bin::Bin,bout::Bout) where {P<:AbstractParticle,Bin<:AbstractBasis,Bout<:AbstractBasis} = Operator(AnnihilationOperator(p),bin,bout)
AnnihilationOperator(p::P,b::B) where {P<:AbstractParticle,B<:AbstractBasis} = AnnihilationOperator(p,b,b)
particles(c::AnnihilationOperator) = c.particles
Base.adjoint(c::AnnihilationOperator) = AnnihilationOperator(c.particles,broadcast(!,c.types))


apply(op::Operator,ind::Integer, bin = preimagebasis(op),bout=imagebasis(op)) = apply(op.op,ind,bin,bout)
apply(op::AnnihilationOperator,ind,bin::B,bout::B) where B<:AbstractBasis = addparticle(particle(op),ind,bin,bout)

# addparticle(f::Fermion, ind,bin,bout) = addparticle(f,ind,bin)
function addparticle(f::Fermion, ind,bin, bout) 
    newstate, newamp = addfermion(siteindex(f,bin), basisstate(ind,bin))
    newind = index(newstate,bout)#,index.(newstates, bout)
    newind, newamp
end
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
        newind, amp = apply(op, ind)
        # for (newind,amp) in zip(newinds,amps)
        state2[newind] += val*amp
        # end
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
    return newfocknbr, allowed * fermionstatistics
end
const BasisOrMissing = Union{AbstractBasis,Missing}
struct OperatorSum{Bin,Bout,T,Ops} <: AbstractOperator{Bin,Bout}
    amplitudes::Vector{T}
    operators::Vector{Ops}
    preimagebasis::Bin
    imagebasis::Bout
    function OperatorSum(amplitudes::Vector{T},ops::Vector{Ops},bin::Bin,bout::Bout) where {Ops,T,Bin<:BasisOrMissing,Bout<:BasisOrMissing}
        newops, newamps = groupbykeysandreduce(ops,amplitudes,+)
        new{Bin,Bout,promote_type(T,eltype.(ops)...),Ops}(newamps,newops,bin,bout)
    end
end

struct OperatorProduct{Ops,Bin,Bout} <: AbstractOperator{Bin,Bout}
    # amplitude::T
    operators::Ops
    preimagebasis::Bin
    imagebasis::Bout
    function OperatorProduct(ops::Ops,bin::Bin,bout::Bout) where {Ops,Bin<:BasisOrMissing,Bout<:BasisOrMissing}
        new{Ops,Bin,Bout}(ops,bin,bout)
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
Base.:-(o1::Union{Operator,OperatorProduct,OperatorSum},o2::Union{Operator,OperatorProduct,OperatorSum}) = OperatorSum(o1) + (-OperatorSum(o2))
function Base.:+(opsum1::OperatorSum,opsum2::OperatorSum)
    newamps = vcat(amplitudes(opsum1),amplitudes(opsum2))
    newops = vcat(operators(opsum1),operators(opsum2))
    newpreimagebasis = promote_basis(preimagebasis(opsum1),preimagebasis(opsum1))
    newimagebasis = promote_basis(imagebasis(opsum1),imagebasis(opsum1))
    OperatorSum(newamps,newops,newpreimagebasis,newimagebasis)
end
Base.:-(opsum::OperatorSum) = OperatorSum(-amplitudes(opsum),operators(opsum),preimagebasis(opsum),imagebasis(opsum))
Base.:-(op::Operator) = -OperatorSum(op)

preimagebasis(op::OperatorProduct) = op.preimagebasis
imagebasis(op::OperatorProduct) = op.imagebasis
amplitude(op::OperatorProduct) = op.amplitude
operators(op::OperatorProduct) = op.operators
Base.eltype(op::OperatorProduct) = promote_type(eltype.(operators(op))...)
OperatorProduct(op::Operator) = OperatorProduct((op,),preimagebasis(op),imagebasis(op))

Base.:*(op::Operator,x::Number) = x*op
Base.:*(x::Number,op::Operator) = x*OperatorSum(op)
Base.:*(x::Number,op::OperatorProduct) = x*OperatorSum(op)
Base.:*(x::Number,op::OperatorSum) = OperatorSum(x.*amplitudes(op),operators(op),preimagebasis(op),imagebasis(op))
Base.:*(op1::Operator,op2::Operator) = OperatorProduct(op1) * OperatorProduct(op2)
Base.:*(op1::OperatorProduct,op2::Operator) = op1 * OperatorProduct(op2)
Base.:*(op1::Operator,op2::OperatorProduct) = OperatorProduct(op1) * op2
function Base.:*(op1::OperatorProduct,op2::OperatorProduct) 
    # newamp = amplitude(op1) * amplitude(op2)
    newops = (operators(op1)...,operators(op2)...)
    OperatorProduct(newops,preimagebasis(op2),imagebasis(op1))
end

function Base.:*(op1::OperatorSum,op2::OperatorSum)
    newops = vec(map(ops->ops[1]*ops[2],Base.product(operators(op1),operators(op2))))
    newamps = vec(map(amps->amps[1]*amps[2],Base.product(amplitudes(op1),amplitudes(op2))))
    OperatorSum(newamps,newops,preimagebasis(op2),imagebasis(op1))
end

# apply(opsum::OperatorSum{<:Any,<:Any,T},ind::Integer,scaling=one(T)) where T = apply(opsum,[ind],[scaling])
# apply(ops::OperatorProduct{<:Any,<:Any,<:Any,T},ind::Integer,scaling=one(T)) where T = apply(ops,(ind,),(scaling,))

# function manyapply(opsum::OperatorSum,inds,scalings)
#     din = Dict(zip(inds,scalings))
#     dout = typeof(din)()
#     for (ind,amp) in pairs(din)
#         for (op,scaling) in pairs(opsum)
#             newstates, amps = apply(op,ind,preimagebasis(op),imagebasis(op))
#             newamps = amps .* (amp*scaling)
#             d2 = Dict(zip(newstates,newamps))
#             mergewith!(+,dout,d2)
#         end
#     end
#     println("ASD")
#     return keysvalues(dout)
# end
function keysvalues(d)
    k = collect(keys(d)) 
    v = map(key->d[key],k) 
    return k,v
end

# function _group(k::K,v::V) where {K,V}
#     d = groupreduce(first,last,+,zip(k,v))
#     ks::K = collect(keys(d))
#     vs::V = collect(values(d))
#     return ks, vs
# end
# groupbykeys(keys,vals) = groupbykeysandreduce(keys,vals)
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
# function groupbyandreduce(iter,f) where {K,V}
#     d = groupreduce(first,last,f,iter)
#     ks::K = collect(keys(d))
#     vs::V = collect(values(d))
#     return ks, vs
# end
# function _merge(k1::K1,v1::V1,k2::K2,v2::V2) where {K1,K2,V1,V2}
#     Knew = promote_type(eltype(K1),eltype(K2))
#     Vnew = promote_type(eltype(V1),eltype(V2))
#     d = groupreduce(first,last,+,Iterators.flatten((zip(k1,v1),zip(k2,v2))))
#     k::Vector{Knew},v::Vector{Vnew} = keysvalues(d)
#     return k, v
# end

function _timetest(N)
    dout = Dict{Integer,Float64}()
    ks = rand(Int,N)
    vs = rand(Float64,N)
    @time for (k,v) in zip(ks,vs)
        newinds = rand(Int,3)
        newamps = rand(Float64,3)
        dnew = Dict(zip(newinds,newamps))
        mergewith!(+,dout,dnew)
    end
    allinds = Int[]
    allamps = Float64[]
    @time for (k,v) in zip(ks,vs)
        newinds = rand(Int,3)
        newamps = rand(Float64,3)
        # push!(allinds,newinds...)
        # push!(allamps,newamps...)
        vcat(allinds,newinds)
        vcat(allamps,newamps)
        # mergewith!(+,dout,dnew)
    end
    allinds,allamps = groupbykeysandreduce(allinds,allamps,+)
end

function apply(opsum::OperatorSum,ind::Integer)
    # dout = Dict{Integer,eltype(opsum)}()
    allinds = typeof(ind)[]
    allamps = eltype(opsum)[]
    for (op,scaling) in pairs(opsum)
        newinds, amps = apply(op,ind,preimagebasis(op),imagebasis(op))
        newamps = amps .* (scaling)
        allinds = vcat(allinds,newinds)
        allamps = vcat(allamps,newamps)
        # dnew = Dict(zip(newinds,newamps))
        # mergewith!(+,dout,dnew)
    end
    return groupbykeysandreduce(allinds,allamps,+)
    # return keysvalues(dout)
end
Base.pairs(opsum::OperatorSum) = zip(operators(opsum),amplitudes(opsum))

function apply(op::OperatorProduct,ind,bin = preimagebasis(op),bout = imagebasis(op))
    function _apply(op,ind,scale)
        newind,newamp = apply(op,ind)
        newind, scale*newamp
    end
    foldr((op,ia) -> _apply(op,ia...),operators(op),init=(ind,one(eltype(op))))
    # manyapply(op,[ind],[one(eltype(op))],bin,bout)
end

# function manyapply(op::Operator,inds,amps,bin = preimagebasis(op),bout = imagebasis(op))
#     allinds = similar(inds)
#     allamps = similar(amps)
#     for (ind,scaling) in zip(inds,amps)
#         newind, newamp = apply(op,ind,bin,bout)
#         push!(allinds,newind)
#         push!(allamps,scaling .* newamp)
#     end
#     newinds, newamps = groupbykeysandreduce(allinds,allamps,+)
#     return newinds,newamps
# end

# function manyapply(op::OperatorProduct,inds,scalings,bin = preimagebasis(op),bout = imagebasis(op))
#     ops = operators(op)
#     scaling = amplitude(op)
#     newind,newamp = foldr((op,is) -> manyapply(op,is...),ops,init=(inds,scalings))
#     println(newinds)
#     println(newamps)
#     println(scaling)
#     return newinds, scaling .* newamps
# end