focknbr(bits::Union{BitVector,Vector{Bool},NTuple{<:Any,Bool}}) = mapreduce(nb -> nb[2] * (1 << (nb[1]-1)) ,+, enumerate(bits))
focknbr(site::Integer) = 1 << (site-1)
focknbr(sites::Vector{<:Integer}) = mapreduce(focknbr,+, sites)
focknbr(sites::NTuple{N,<:Integer}) where N = mapreduce(site -> 1 << (site-1),+, sites)

bits(s::Integer,N) = digits(Bool,s, base=2, pad=N)
parity(fs::Int) = iseven(fermionnumber(fs)) ? 1 : -1
fermionnumber(fs::Int) = count_ones(fs)

siteindex(id,b::AbstractBasis) = findfirst(x->x==id, labels(b))::Int
siteindices(ids::Union{Tuple{S,Vararg{S}}, AbstractVector{S}}, b::AbstractBasis) where {S} = map(id->siteindex(id,b),ids)#::Int

function tensor(v::AbstractVector{T}, b::FermionBasis{M}) where {T,M}
    @assert length(v) == 2^M
    t = Array{T,M}(undef,ntuple(i->2,M))
    for I in CartesianIndices(t)
        fs = focknbr(Bool.(Tuple(I) .- 1))
        t[I] = v[focktoind(fs,b)] #* parity(fs)
    end
    return t
end
##https://iopscience.iop.org/article/10.1088/1751-8121/ac0646/pdf (10c)
# _bit(f,k) = Bool(sign(f & 2^(k-1)))
_bit(f,k) = Bool((f >> (k-1)) & 1)
function phase_factor(focknbr1,focknbr2,subinds::NTuple) 
    bitmask = focknbr(subinds)
    prod(i-> (jwstring(i, bitmask & focknbr1)*jwstring(i, bitmask & focknbr2))^_bit(focknbr2,i),subinds)
end
function phase_factor(focknbr1,focknbr2,::Val{N}) where N
    prod(ntuple(i-> phase_factor(focknbr1,focknbr2,i),N))
end

function phase_factor(focknbr1,focknbr2,i::Integer)
    _bit(focknbr2,i) ? (jwstring(i, focknbr1)*jwstring(i, focknbr2)) : 1
end

reduced_density_matrix(v::AbstractMatrix, bsub::FermionBasis, bfull::FermionBasis) = reduced_density_matrix(v,Tuple(keys(bsub)),bfull, bsub.symmetry)

reduced_density_matrix(v::AbstractVector, args...) = reduced_density_matrix(v*v',args...)
function reduced_density_matrix(m::AbstractMatrix{T}, labels::NTuple{N}, b::FermionBasis{M}, sym::AbstractSymmetry = NoSymmetry()) where {N,T,M}
    mout = zeros(T,2^N,2^N)
    reduced_density_matrix!(mout,m,labels,b,sym)
end
function reduced_density_matrix!(mout,m::AbstractMatrix{T}, labels::NTuple{N}, b::FermionBasis{M}, sym::AbstractSymmetry = NoSymmetry()) where {N,T,M}
    mout .*= 0
    outinds::NTuple{N,Int} = siteindices(labels, b)
    @assert all(diff([outinds...]) .> 0) "Subsystems must be ordered in the same way as the full system"
    bitmask = 2^M - 1 - focknbr(outinds)
    outbits(f) = map(i->_bit(f,i),outinds)
    for f1 in UnitRange{UInt64}(0,2^M-1), f2 in UnitRange{UInt64}(0,2^M-1)
        if (f1 & bitmask) != (f2 & bitmask)
            continue
        end
        newfocknbr1 = focknbr(outbits(f1))
        newfocknbr2 = focknbr(outbits(f2))
        s1 = phase_factor(f1,f2,Val(M))
        s2 = phase_factor(newfocknbr1,newfocknbr2, Val(N))
        s = s2*s1
        mout[focktoind(newfocknbr1, sym), focktoind(newfocknbr2, sym)] += s*m[focktoind(f1,b),focktoind(f2,b)]#*s2
    end
    return mout
end

function Base.Matrix(t::AbstractArray{<:Any,N}, leftindices::NTuple{NL,Int}) where {N,NL}
    rightindices::NTuple{N-NL,Int} = Tuple(setdiff(ntuple(identity,N), leftindices))
    Matrix(t,leftindices,rightindices)
end
function Base.Matrix(t::AbstractArray{<:Any,N}, leftindices::NTuple{NL,Int}, rightindices::NTuple{NR,Int}) where {N,NL,NR}
    @assert NL+NR == N
    tperm = permutedims(t,(leftindices...,rightindices...))
    lsize = prod(i->size(tperm,i), leftindices, init=1)
    rsize = prod(i->size(tperm,i), rightindices, init=1)
    reshape(tperm, lsize, rsize) 
end

function LinearAlgebra.svd(v::AbstractVector, leftlabels::NTuple{N}, b::FermionBasis{M}) where {N,M}
    linds = siteindices(leftlabels, b)
    t = tensor(v,b)
    svd(Matrix(t,linds))
end