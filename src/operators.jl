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
operator(op::FockOperator) = op.op

preimagebasis(::AbstractFockOperator{Missing}) = missing
imagebasis(::AbstractFockOperator{<:Any,Missing}) = missing
preimagebasis(::Fermion) = missing
imagebasis(::Fermion) = missing
Base.eltype(::CreationOperator) = Int
CreationOperator(f::Fermion) = CreationOperator((f,),(true,))
AnnihilationOperator(f::Fermion) = CreationOperator(f)'
Base.:*(f1::Fermion,f2::Fermion) = AnnihilationOperator(f1)*AnnihilationOperator(f2)
# Base.:*(f1::Fermion,f2::CreationOperator) = AnnihilationOperator(f1)*f2
# Base.:*(f1::CreationOperator,f2::Fermion) = f1*AnnihilationOperator(f2)

Base.:+(f1::Fermion,f2::Fermion) = AnnihilationOperator(f1) + AnnihilationOperator(f2)
Base.:+(f1::Fermion,f2::CreationOperator) = AnnihilationOperator(f1) + f2
Base.:+(f1::CreationOperator,f2::Fermion) = f1 + AnnihilationOperator(f2)

Base.:*(c1::CreationOperator,c2::CreationOperator) = CreationOperator((particles(c2)...,particles(c1)...),(c2.types...,c1.types...))
# Base.:*(c1::FockOperator{<:Any,<:Any,<:CreationOperator},c2::FockOperator{<:Any,<:Any,<:CreationOperator}) = FockOperator(c1.op*c2.op,preimagebasis(c2),imagebasis(c1))
# Base.:*(c1::Union{CreationOperator,Fermion},c2::FockOperator{<:Any,<:Any,<:CreationOperator}) = FockOperator(c1*c2.op,preimagebasis(c2),imagebasis(c2))
# Base.:*(c1::FockOperator{<:Any,<:Any,<:CreationOperator},c2::Union{CreationOperator,Fermion}) = FockOperator(c1.op*c2,preimagebasis(c1),imagebasis(c1))

# Base.:*(c1::CreationOperator,c2::FockOperatorSum) = FockOperator(c1)*c2
# Base.:*(c1::FockOperatorSum,c2::CreationOperator) = c1*FockOperator(c2)


FockOperator(c::CreationOperator) = FockOperator(c,missing,missing)

Base.:+(c1::CreationOperator,c2::CreationOperator) = FockOperator(c1,missing,missing) + FockOperator(c2,missing,missing)
Base.:+(c1::CreationOperator,c2::Union{FockOperator,FockOperatorSum}) = FockOperator(c1,preimagebasis(c2),imagebasis(c2)) + c2
Base.:+(f::Fermion,c::Union{FockOperator,FockOperatorSum}) = AnnihilationOperator(f) + c
Base.:+(c::Union{FockOperator,FockOperatorSum},f::Fermion) = c + AnnihilationOperator(f) 
Base.:+(c1::Union{FockOperator,FockOperatorSum},c2::CreationOperator) = c1 + FockOperator(c2,preimagebasis(c1),imagebasis(c1))
Base.:-(c1::CreationOperator,c2::CreationOperator) = FockOperator(c1,missing,missing) - FockOperator(c2,missing,missing)
Base.:-(c1::Union{CreationOperator,Fermion},c2::Union{FockOperator,FockOperatorSum}) = FockOperator(c1,preimagebasis(c2),imagebasis(c2)) - c2
Base.:-(c1::Union{FockOperator,FockOperatorSum},c2::Union{CreationOperator,Fermion}) = c1 - FockOperator(c2,preimagebasis(c1),imagebasis(c1))

function Base.:*(c1::FockOperator{<:Any,<:Any,<:CreationOperator},c2::FockOperatorSum)
    bin = promote_basis(preimagebasis(c1),preimagebasis(c2))
    bout = promote_basis(imagebasis(c1),imagebasis(c2))
    FockOperatorSum(amplitudes(c2),[operator(c1) * op for op in operators(c2)] ,bin,bout)
end
function Base.:*(c1::FockOperatorSum,c2::FockOperator{<:Any,<:Any,<:CreationOperator})
    bin = promote_basis(preimagebasis(c1),preimagebasis(c2))
    bout = promote_basis(imagebasis(c1),imagebasis(c2))
    FockOperatorSum(amplitudes(c1), [op * operator(c2) for op in operators(c1)],bin,bout)
end

FermionCreationOperator(id,bin::Bin,bout::Bout) where {Bin<:AbstractBasis,Bout<:AbstractBasis} = CreationOperator(Fermion(id),bin,bout)
FermionCreationOperator(id,b::B) where B<:AbstractBasis = FermionCreationOperator(id,b,b)
CreationOperator(p::P,bin::Bin,bout::Bout) where {P<:AbstractParticle,Bin<:AbstractBasis,Bout<:AbstractBasis} = FockOperator(CreationOperator(p),bin,bout)
CreationOperator(p::P,b::B) where {P<:AbstractParticle,B<:AbstractBasis} = CreationOperator(p,b,b)
particles(c::CreationOperator) = c.particles
Base.adjoint(c::CreationOperator) = CreationOperator(reverse(c.particles),broadcast(!,reverse(c.types)))
Base.adjoint(op::FockOperator) = FockOperator(operator(op)',imagebasis(op),preimagebasis(op))

FockOperatorSum(op::FockOperatorSum,b::BasisOrMissing) = FockOperatorSum(amplitudes(op),operators(op),b,b)

apply(op::FockOperator,ind::Integer, bin = preimagebasis(op),bout=imagebasis(op)) = apply(op.op,ind,bin,bout)

siteindices(ps,bin) = map(p->siteindex(p,bin),ps)
function apply(op::CreationOperator{<:Fermion}, ind,bin, bout)
    newstate, newamp = togglefermions(siteindices(particles(op),bin),op.types,basisstate(ind,bin))
    newind = index(newstate,bout)
    newind, newamp
end


index(basisstate::Integer,::FermionBasis) = basisstate + 1
basisstate(ind::Integer,::FermionBasis) = ind - 1

function Base.:*(op::AbstractFockOperator, state::AbstractVector)
    out = zero(state)
    mul!(out,op,state)
end
function Base.:*(state::Adjoint{<:Any,<:AbstractVector},op::AbstractFockOperator)
    out = zero(state')
    mul!(out,op',state')'
end
function LinearAlgebra.mul!(state2,op::AbstractFockOperator, state)
    state2  .= zero(state2)
    bin = promote_basis(preimagebasis(op),basis(state))
    bout = promote_basis(imagebasis(op),basis(state2))
    for (ind,val) in pairs(state)
        newind, amp = apply(op, ind,bin,bout)
        state2[newind] += val*amp
    end
    return state2
end

function LinearAlgebra.mul!(state2,ops::FockOperatorSum, state)
    state2 .= zero(state2)
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
# Base.Matrix(op::AbstractFockOperator,basis) = 

preimagebasis(op::FockOperatorSum) = op.preimagebasis
imagebasis(op::FockOperatorSum) = op.imagebasis
amplitudes(op::FockOperatorSum) = op.amplitudes
operators(op::FockOperatorSum) = op.operators
Base.eltype(::FockOperatorSum{<:Any,<:Any,T}) where T = T
FockOperatorSum(op::FockOperator) = FockOperatorSum([one(eltype(op))],[operator(op)],preimagebasis(op),imagebasis(op))
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

FockOperator(op::AbstractElementaryFockOperator,b::BasisOrMissing) = FockOperator(op,b,b)

Base.:*(x::Number,f::Fermion) = x*FockOperatorSum(f)
FockOperatorSum(c::AbstractElementaryFockOperator) = FockOperatorSum([one(eltype(c))],[c],missing,missing)
FockOperator(f::Fermion,args...) = FockOperator(AnnihilationOperator(f),args...)
FockOperatorSum(f::Fermion) = FockOperatorSum(AnnihilationOperator(f))
Base.:*(x::Number,op::Union{FockOperator,FockOperatorProduct,AbstractElementaryFockOperator}) = x*FockOperatorSum(op)
Base.:*(op::FockOperator,x::Number) = x*op
# Base.:*(x::Number,op::FockOperator) = x*FockOperatorSum(op)
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

Base.:*(f::Fermion,op::AbstractFockOperator) = AnnihilationOperator(f)*op
Base.:*(op::AbstractFockOperator,f::Fermion) = op*AnnihilationOperator(f)

FockOperatorSum(amps,ops) = FockOperatorSum(amps,ops,preimagebasis(last(ops)),imagebasis(first(ops))) 

Base.:*(b::BasisOrMissing,op::Union{AbstractElementaryFockOperator,Fermion}) = FockOperator(op,preimagebasis(op),b)
Base.:*(op::Union{AbstractElementaryFockOperator,Fermion},b::BasisOrMissing) = FockOperator(op,b,imagebasis(op))
Base.:*(b::BasisOrMissing,op::FockOperator) = FockOperator(operator(op),preimagebasis(op),b)
Base.:*(op::FockOperator,b::BasisOrMissing) = FockOperator(operator(op),b,imagebasis(op))
Base.:*(b::BasisOrMissing,op::FockOperatorProduct) = FockOperatorProduct(operators(op),preimagebasis(op),b)
Base.:*(op::FockOperatorProduct,b::BasisOrMissing) = FockOperatorProduct(operators(op),b,imagebasis(op))
Base.:*(b::BasisOrMissing,op::FockOperatorSum) = FockOperatorSum(amplitudes(op),operators(op),preimagebasis(op),b)
Base.:*(op::FockOperatorSum,b::BasisOrMissing) = FockOperatorSum(amplitudes(op),operators(op),b,imagebasis(op))

Base.:*(c::AbstractElementaryFockOperator,op::FockOperator) = imagebasis(op)*(c*operator(op))*preimagebasis(op)
Base.:*(op::FockOperator,c::AbstractElementaryFockOperator) = imagebasis(op)*(operator(op)*c)*preimagebasis(op)#FockOperator(operator(op)*c,preimagebasis(op),imagebasis(op))
Base.:*(op1::FockOperator,op2::FockOperator) = FockOperator(operator(op1)*operator(op2),preimagebasis(op2),imagebasis(op1))
Base.:*(op1::FockOperatorProduct,op2::FockOperator) = op1 * FockOperatorProduct(op2)
Base.:*(op1::FockOperator,op2::FockOperatorProduct) = FockOperatorProduct(op1) * op2
function Base.:*(op1::FockOperatorProduct,op2::FockOperatorProduct) 
    newops = (operators(op1)...,operators(op2)...)
    FockOperatorProduct(newops,preimagebasis(op2),imagebasis(op1))
end
Base.:*(op1::AbstractFockOperator,op2::AbstractFockOperator) = FockOperatorProduct(op1)*FockOperatorProduct(op2)

function Base.:*(op::AbstractFockOperator,sum::FockOperatorSum)
    newops = [op*sop for sop in operators(sum)]
    FockOperatorSum(amplitudes(sum),newops)
end
function Base.:*(sum::FockOperatorSum,op::AbstractFockOperator)
    newops = [sop*op for sop in operators(sum)]
    FockOperatorSum(amplitudes(sum),newops)
end

##Parity 
struct ParityOperator <: AbstractElementaryFockOperator{Missing,Missing} end
Base.adjoint(::ParityOperator) = ParityOperator()
Base.eltype(::ParityOperator) = Int
parity(fs::Int) = (-1)^count_ones(fs)
function apply(op::ParityOperator,ind::Integer, bin = preimagebasis(op),bout=imagebasis(op))
    focknbr = basisstate(ind,bin)
    return index(focknbr,bout), parity(focknbr)
end



function LinearAlgebra.Matrix(ops::FockOperatorSum)
    bin = preimagebasis(ops)
    bout = imagebasis(ops)
    mat = zeros(eltype(ops),length(bout),length(bin))
    for (op,opamp) in pairs(ops)
        for ind in eachindex(bin)
            newind, amp = apply(op, ind,bin,bout)
            mat[newind,ind] +=  opamp*amp
        end
    end
    return mat
end
function SparseArrays.sparse(ops::FockOperatorSum)
    bin = preimagebasis(ops)
    bout = imagebasis(ops)
    mat = spzeros(eltype(ops),length(bout),length(bin))
    for (op,opamp) in pairs(ops)
        for ind in eachindex(bin)
            newind, amp = apply(op, ind,bin,bout)
            mat[newind,ind] += opamp*amp
        end
    end
    return mat
end
LinearAlgebra.Matrix(op::AbstractFockOperator) = Matrix(FockOperatorSum(op))
SparseArrays.sparse(op::AbstractFockOperator) = sparse(FockOperatorSum(op))

function inner(w,ops::FockOperatorSum,v)
    bin = preimagebasis(ops)
    bout = imagebasis(ops)
    res = zero(eltype(ops))
    for (op,opamp) in pairs(ops)
        for (ind,vamp) in pairs(v)
            newind, amp = apply(op, ind,bin,bout)
            res += opamp*amp*vamp*w[newind]
        end
    end
    res
end

function measure(ops::FockOperatorSum,v)
    bin = preimagebasis(ops)
    bout = imagebasis(ops)
    res = zero(eltype(ops))
    for (op,opamp) in pairs(ops)
        for (ind,vamp) in pairs(v)
            newind, amp = apply(op, ind,bin,bout)
            res += opamp*amp*vamp*v[newind]'
        end
    end
    res
end